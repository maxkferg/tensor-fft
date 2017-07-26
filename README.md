# Tensorflow FFT
An efficient implementation of Welches method for calculating periodograms


## Welches Method
A massively parrallel implimentation of Welches method with support for batching and GPU execution.
Implimented with Tensorflow.

```python
from core.welches import WelchProcessor

with tf.Session() as sess:
	window_length = 1000
	signal_length = 1000

	t = np.linspace(0, 50, signal_length)
	signal = np.sin(math.pi*t)

	welch = WelchProcessor(sess, "hann", window_length=window_length, max_signal_length=signal_length)
	
	frequencies = np.linspace(0, 10, welch.output_length)
	amplitudes = welch.feed(signal)

	# Plot the periodograms
	plt.plot(frequencies, amplitudes)
```

## Serving
TODO



## License
MIT