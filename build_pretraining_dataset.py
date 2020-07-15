# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Writes out text data as tfrecords that ELECTRA can be pre-trained on."""

import functools
import signal
import threading
from multiprocessing import Manager, Queue, Pool
import concurrent.futures
import argparse
import multiprocessing
import os
import random
import time
import queue
import logging

import tensorflow.compat.v1 as tf
import humanize

from transformers import BertTokenizerFast
#from model import tokenization
from util import utils

LOGLEVEL = os.environ.get('LOGLEVEL', 'WARNING').upper()
logging.basicConfig(level=LOGLEVEL)

def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


class ExampleBuilder(object):
    """Given a stream of input text, creates pretraining examples."""

    def __init__(self, tokenizer, max_length):
        self._tokenizer = tokenizer
        self._current_sentences = []
        self._current_length = 0
        self._max_length = max_length
        self._target_length = max_length

    def add_line(self, line):
        """Adds a line of text to the current example being built."""
        line = line.strip().replace("\n", " ")
        if (not line) and self._current_length != 0:  # empty lines separate docs
            return self._create_example()
        bert_tokens = self._tokenizer.tokenize(line)
        bert_tokids = self._tokenizer.convert_tokens_to_ids(bert_tokens)
        self._current_sentences.append(bert_tokids)
        self._current_length += len(bert_tokids)
        if self._current_length >= self._target_length:
            return self._create_example()
        return None

    def _create_example(self):
        """Creates a pre-training example from the current list of sentences."""
        # small chance to only have one segment as in classification tasks
        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (self._target_length - 3) // 2

        first_segment = []
        second_segment = []
        for sentence in self._current_sentences:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (len(first_segment) == 0 or
                len(first_segment) + len(sentence) < first_segment_target_length or
                (len(second_segment) == 0 and
                 len(first_segment) < first_segment_target_length and
                 random.random() < 0.5)):
                first_segment += sentence
            else:
                second_segment += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[:self._max_length - 2]
        second_segment = second_segment[:max(0, self._max_length -
                                             len(first_segment) - 3)]

        # prepare to start building the next example
        self._current_sentences = []
        self._current_length = 0
        # small chance for random-length instead of max_length-length example
        if random.random() < 0.05:
            self._target_length = random.randint(5, self._max_length)
        else:
            self._target_length = self._max_length

        return self._make_tf_example(first_segment, second_segment)

    def _make_tf_example(self, first_segment, second_segment):
        """Converts two "segments" of text into a tf.train.Example."""
        SEP = self._tokenizer.sep_token_id
        CLS = self._tokenizer.cls_token_id
        input_ids = [CLS] + first_segment + [SEP]
        segment_ids = [0] * len(input_ids)
        if second_segment:
            input_ids += second_segment + [SEP]
            segment_ids += [1] * (len(second_segment) + 1)
        input_mask = [1] * len(input_ids)
        input_ids += [0] * (self._max_length - len(input_ids))
        input_mask += [0] * (self._max_length - len(input_mask))
        segment_ids += [0] * (self._max_length - len(segment_ids))
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            "input_ids": create_int_feature(input_ids),
            "input_mask": create_int_feature(input_mask),
            "segment_ids": create_int_feature(segment_ids)
        }))
        return tf_example

class ExampleWriter(object):
    """Writes pre-training examples to disk."""

    def __init__(self, output_fname, vocab_file, max_seq_length,
                 blanks_separate_docs, do_lower_case):
        self._blanks_separate_docs = blanks_separate_docs
        tokenizer = BertTokenizerFast(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case)
        self._example_builder = ExampleBuilder(tokenizer, max_seq_length)
        self._writers = []
        self._wd = tf.io.TFRecordWriter(output_fname)
        self.n_written = 0

    def write(self, example):
        if not example:
            return
        self._wd.write(example.SerializeToString())
        self.n_written += 1

    def covert(self, input_file):
        """Writes out examples from the provided input file."""
        with tf.io.gfile.GFile(input_file) as f:
            for line in f:
                line = line.strip()
                if line or self._blanks_separate_docs:
                    self.write(self._example_builder.add_line(line))
            self.write(self._example_builder.add_line(""))

    def finish(self):
        for writer in self._writers:
            writer.close()


def distribute(args):
    fnames = sorted(tf.io.gfile.listdir(args.corpus_dir))
    jobs = []
    with concurrent.futures.ProcessPool(args.num_processes) as executor:
        random.shuffle(fnames)

        for (i, f) in enumerate(fnames):
            future = executor.submit(write_examples, f)
            jobs.append(future)

    total = 0
    for f in tqdm.tqdm(futures):
        f.result()


def write_examples(filenames):
    """A single process creating and writing out pre-processed examples."""

    output_fname = os.path.join(
        output_dir, "pretrain_data.tfrecord-{:%04d}-of-{:%04d}".format(i, num_out_files))

    example_writer = ExampleWriter(
        job_id=job_id,
        vocab_file=args.vocab_file,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        num_jobs=args.num_processes,
        blanks_separate_docs=args.blanks_separate_docs,
        do_lower_case=args.do_lower_case
    )
    log("Writing tf examples")
    fnames = sorted(tf.io.gfile.listdir(args.corpus_dir))
    fnames = [f for (i, f) in enumerate(fnames)
              if i % args.num_processes == job_id]
    random.shuffle(fnames)
    start_time = time.time()
    for file_no, fname in enumerate(fnames):
        if file_no > 0:
            elapsed = time.time() - start_time
            log("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ETA: {:}s, "
                "{:} examples written".format(
                    file_no, len(fnames), 100.0 * file_no /
                    len(fnames), int(elapsed),
                    int((len(fnames) - file_no) / (file_no / elapsed)),
                    example_writer.n_written))
        example_writer.write_examples(os.path.join(args.corpus_dir, fname))
    example_writer.finish()
    log("Done!")


shutdown_event = threading.Event()


class Counter(object):
    def __init__(self):
        self.val = multiprocessing.Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n
        return self.val.value

    @property
    def value(self):
        return self.val.value


def example_writer(output_fname_template: str, counter: Counter, message_queue: Queue):
    output_fname = output_fname_template.format(counter.increment())
    Z_NO_FLUSH = 0
    Z_PARTIAL_FLUSH = 1
    Z_SYNC_FLUSH = 2
    Z_FULL_FLUSH = 3
    options = tf.io.TFRecordOptions(
        compression_type='ZLIB', flush_mode=Z_NO_FLUSH, 
        input_buffer_size=None,
        output_buffer_size=None, window_bits=None, compression_level=None,
        compression_method=None, mem_level=None, compression_strategy=None
    )
    total = 0
    total_bytes = 0
    fd = tf.io.TFRecordWriter(output_fname, options=options)
    start = time.time()
    logger = logging.getLogger()
    while not shutdown_event.is_set():
        try:
            example = message_queue.get(block=True, timeout=0.05)
            fd.write(example)
            total_bytes += len(example) 
            total += 1
            logger.info('wrote %1.f examples/sec', total / (time.time()-start))
            logger.info('wrote %s bytes/sec', humanize.filesize.naturalsize(total_bytes / (time.time()-start)))
            logger.info('wrote %s total bytes', humanize.filesize.naturalsize(total_bytes))

            if total_bytes > 200e6:
                fd.close()
                output_fname = output_fname_template.format(
                    counter.increment())
                fd = tf.io.TFRecordWriter(output_fname, options=options)
                total_bytes = 0
        except queue.Empty:
            logger.debug('nothing to write ...')
            time.sleep(0.01)
            continue
        except ValueError:
            # queue close
            fd.close()
            break
    return total

class TFWriter(object):

    def __init__(self, max_byte_size: int, file_template: str, file_counter: Counter):
        self._file_templates = file_template
        self._file_counter = file_counter
        self._fd = self.open_new_file()
        self._total = 0
        self._total_bytes = 0
        self._max_byte_size_per_file = max_byte_size

    def open_new_file(self):
        Z_NO_FLUSH = 0
        Z_PARTIAL_FLUSH = 1
        Z_SYNC_FLUSH = 2
        Z_FULL_FLUSH = 3
        options = tf.io.TFRecordOptions(
            compression_type='ZLIB', flush_mode=Z_NO_FLUSH, 
            input_buffer_size=None,
            output_buffer_size=None, window_bits=None, compression_level=None,
            compression_method=None, mem_level=None, compression_strategy=None
        )
        output_fname = self._file_templates.format(self._file_counter.increment())
        return tf.io.TFRecordWriter(output_fname, options=options)

    def _reset_stats(self):
        self._total = 0
        self._total_bytes = 0

    def write(self, example):
        self._fd.write(example)
        self._total_bytes += len(example) 
        self._total += 1
        if self._total_bytes > self._max_byte_size_per_file:
            self._fd.close()
            self._fd = self.open_new_file()
            self._reset_stats()

    def close(self):
        self._fd.close() 

def line_tokenizer_reader(args, file_queue: Queue, line_counter: Counter, file_counter: Counter):
    blanks_separate_docs = args.blanks_separate_docs

    # tokenizer = tokenization.FullTokenizer(
    #     vocab_file=args.vocab_file,
    #     do_lower_case=args.do_lower_case)
    # tokenizer = BertTokenizerFast(vocab_file=args.vocab_file, 
    #                           do_lower_case=args.do_lower_case, 
    #                           unk_token='[UNK]', 
    #                           tokenize_chinese_chars=False, 
    #                           wordpieces_prefix='##')
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_file,
                          do_lower_case=args.do_lower_case, 
                          unk_token='[UNK]', 
                          sep_token='[SEP]',
                          strip_accents=False,
                          clean_text=True,
                          tokenize_chinese_chars=False, 
                          wordpieces_prefix='##')

    example_builder = ExampleBuilder(tokenizer, args.max_seq_length)
    output_filename_template = os.path.join(
        args.output_dir, "pretrain_data-{:04d}.tfrecord.lz")

    sent_messages = 0
    logger = logging.getLogger()

    TWO_HUNDREND_MB = 400e6
    writer = TFWriter(TWO_HUNDREND_MB, output_filename_template, file_counter)

    def send(example):
        if not example:
            return 0
        while not shutdown_event.is_set():
            try:
                #message_queue.put(example.SerializeToString(), block=True, timeout=0.05)
                writer.write(example.SerializeToString())
                break
            except queue.Full:
                logger.warning('queue is full')
                time.sleep(0.01)
                continue
        writer.close()
        return 1
    while not shutdown_event.is_set():
        try:
            input_file = file_queue.get(block=True, timeout=0.05)
        except EOFError:
            continue
        except queue.Empty:
            continue
        except ValueError:
            break
        try:
            lines_read = 0
            start = time.time()
            bytes_read = 0
            already_sent = sent_messages
            previous_reading = 0
            with tf.io.gfile.GFile(input_file) as f:
                for line in f:
                    bytes_read += len(line)
                    line = line.strip()
                    if line or blanks_separate_docs:
                        sent_messages += send(example_builder.add_line(line))
                        lines_read += 1
                        elapsed = time.time()
                        if lines_read % 100:
                            line_counter.increment(lines_read - previous_reading)
                            previous_reading = lines_read
                        logger.info('reading %.1f lines/sec', lines_read / (elapsed - start))
                        logger.info('read  %s bytes', humanize.filesize.naturalsize(bytes_read))
                        logger.info('sending %.1f messages/sec', (sent_messages - already_sent)/ (elapsed - start))
                sent_messages += send(example_builder.add_line(""))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            logger.error('reading file %r %s', exc, input_file)
    return sent_messages

def log_thread(line_counter):
    from tqdm import auto as tqdm
    breathe_total=30554375
    pbar = tqdm.tqdm(total=breathe_total)
    while not shutdown_event.is_set():
        pbar.n = line_counter.value
        pbar.refresh()
        time.sleep(2)

def distribute(args):
    # filenames, num_writers, num_readers, output_filename_template, output_file_size, compressed=False
    os.makedirs(args.output_dir)


    logger = logging.getLogger()
    with multiprocessing.Manager() as manager:
        cpus = multiprocessing.cpu_count()-1 or 1
        num_writers = cpus
        #num_readers = cpus - num_writers
        # By default pool will size depending on cores available
        #writer_pool = Pool(num_writers)
        # By default pool will size depending on cores available
        #reader_pool = Pool(num_readers)
        file_queue = Queue(maxsize=1)
        #message_queue = Queue(maxsize=num_readers*8)
        file_counter = Counter()
        line_counter = Counter()

        # Start file listener ahead of doing the work
        # writers = []
        # for i in range(num_writers):
        #     w = multiprocessing.Process(target=example_writer,
        #                                 args=(output_filename_template, counter, message_queue))
        #     w.start()
        #     writers.append(w)

        readers = []
        for i in range(num_writers):
            r = multiprocessing.Process(target=line_tokenizer_reader,
                                        args=(args, file_queue, line_counter, file_counter))
            r.start()
            readers.append(r)

        log = multiprocessing.Process(target=log_thread, args=(line_counter,))
        log.start()

        fnames = sorted(tf.io.gfile.listdir(args.corpus_dir))
        random.shuffle(fnames)
        for f in fnames:
            file_queue.put(os.path.join(args.corpus_dir, f))
        logger.debug('waiting for readers to finish...')
        log_thread(line_counter)
        file_queue.close()

        for r in readers:
            r.join()
        # message_queue.join()
        # for w in writers:
        #     w.join()
        logger.debug('done')
        log.join()

def signal_handler(a, b):
    print('received shutdown signal', a, b)
    shutdown_event.set()


def handle_signal(signal_num):
    signal.signal(signal_num, signal_handler)
    signal.siginterrupt(signal_num, False)


def main():
    handle_signal(signal.SIGINT)
    handle_signal(signal.SIGTERM)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True,
                        help="Location of pre-training text files.")
    parser.add_argument("--vocab-file", required=True,
                        help="Location of vocabulary file.")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write out the tfrecords.")
    parser.add_argument("--max-seq-length", default=128, type=int,
                        help="Number of tokens per example.")
    parser.add_argument("--num-processes", default=1, type=int,
                        help="Parallelize across multiple processes.")
    parser.add_argument("--blanks-separate-docs", default=True, type=bool,
                        help="Whether blank lines indicate document boundaries.")
    parser.add_argument("--do-lower-case", dest='do_lower_case',
                        action='store_true', help="Lower case input text.")
    parser.add_argument("--no-lower-case", dest='do_lower_case',
                        action='store_false', help="Don't lower case input text.")
    parser.set_defaults(do_lower_case=True)
    args = parser.parse_args()

    assert not os.path.exists(args.output_dir), args.output_dir

    distribute(args)
    # if args.num_processes == 1:
    #   write_examples(0, args)
    # else:
    #   jobs = []
    #   group_files = [ f for f in enumerate() ]
    #   with multiprocessing.Pool(args.num_processes) as pool:
    #     pool.imap()
    #   for i in range(args.num_processes):
    #     job = multiprocessing.Process(target=write_examples, args=(i, args))
    #     jobs.append(job)
    #     job.start()
    #   for job in jobs:
    #     job.join()


if __name__ == "__main__":
    main()
