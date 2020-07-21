import time
from datetime import datetime
import tensorflow as tf

class LoggerHook(tf.train.SessionRunHook):
  """Logs loss and runtime."""

  def begin(self):
    self._step = -1
    self._start_time = time.time()

  def before_run(self, run_context):
    self._step += 1

  def after_run(self, run_context, run_values):
    log_frequency = 10
    batch_size = 10
    if self._step % log_frequency == 0:
      current_time = time.time()
      duration = current_time - self._start_time
      self._start_time = current_time

      loss_value = run_values.results
      examples_per_sec = log_frequency * batch_size / duration
      sec_per_batch = float(duration / log_frequency)

      format_str = ('%s: step %d (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print (format_str % (datetime.now(), self._step, 
                           examples_per_sec, sec_per_batch))
