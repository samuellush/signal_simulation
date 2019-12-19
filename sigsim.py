import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig

# all unitless distances assumed to be meters throughout project

class Source:
	def __init__(self, xpos, ypos, on=True):
		self.xpos = xpos
		self.ypos = ypos
		self.on = on
		self.signal = np.zeros(0)

	def set_on(self, on=True):
		self.on = on

	def set_off(self, off=True):
		self.on = not off

	def set_signal(self, signal):
		self.signal = signal


class Receiver:
	def __init__(self, name, xpos, ypos):
		self.name = name
		self.xpos = xpos
		self.ypos = ypos
		self.data = np.zeros(0)

	def add_signal(self, signal):
		if len(signal) > len(self.data):
			signal[:len(self.data)] += self.data
			self.data = signal
		else:
			self.data[:len(signal)] += signal

	def clear_data(self):
		self.data = np.zeros(0)

	def get_data(self):
		return self.data


# create circle of N evenly-spaced sources
# angle w along circle for source with index i: w = i * 2*np.pi/N
def create_sources(radius, N, on=True):
	sources = []
	for i in range(N):
		xpos = radius * np.cos(i * 2*np.pi/N)
		ypos = radius * np.sin(i * 2*np.pi/N)
		sources.append(Source(xpos, ypos, on=on))
	return sources

# Euclidean distance between source and receiver
def dist(src, rec):
	dx = src.xpos - rec.xpos
	dy = src.ypos - rec.ypos
	return np.sqrt(dx**2 + dy**2)

def plot_map(sources, receivers):
	fig1 = plt.figure(figsize = (6.0, 6.0))
	axes1 = fig1.add_axes([0.1,0.1,0.8,0.8])
	axes1.plot([src.xpos for src in sources if src.on], [src.ypos for src in sources if src.on], 'g*', label='Enabled Source')
	axes1.plot([src.xpos for src in sources if not src.on], [src.ypos for src in sources if not src.on], 'r*', label='Disabled Source')
	axes1.plot([rec.xpos for rec in receivers], [rec.ypos for rec in receivers], 'v', label='Receiver')
	axes1.set_title('Position of Sources and Receivers')
	[axes1.annotate(rec.name, (rec.xpos, rec.ypos + 0.25)) for rec in receivers]
	fig1.legend(loc = 'lower right')

def plot_receiver_signal(times, receiver):
	signal = receiver.get_data()
	if len(signal) > len(times):
		signal = signal[:len(times)]
	fig1 = plt.figure()
	axes1 = fig1.add_axes([0.1,0.1,0.8,0.8])
	axes1.plot(times, signal)
	axes1.set_title('Signal Received by Receiver ' + receiver.name)

# delay the signal as it travels from source to receiver at constant speed
def travel_delay(dist, v, samplerate, signal):
	t = dist / v
	delay = np.floor(t * samplerate).astype(int)
	return np.concatenate((np.zeros(delay), signal))

# signal from each enabled source is added (with travel delay) to each receiver
def record_data(sources, receivers, v_signal, samplerate, eq_time=False):
	for rec in receivers:
		for src in sources:
			if src.on:
				sig = src.signal
				sig_delayed = travel_delay(dist(rec, src), v_signal, samplerate, sig)
				rec.add_signal(sig_delayed)
	# make each receiver have the same length of time for data capture,
	# by cutting off signals received past the length of the shortest capture
	if eq_time:
		l_shortest = min([len(rec.data) for rec in receivers])
		for rec in receivers:
			rec.data = rec.data[:l_shortest]

def clear_receiver_data(receivers):
	for rec in receivers:
		rec.clear_data()

def correlate(rec1, rec2):
	corr = spsig.correlate(rec1.get_data(), rec2.get_data(), mode='full')
	return (corr, rec1.name + rec2.name)

def plot_correlation(correlation):
	corr = correlation[0]
	name = correlation[1]
	fig1 = plt.figure()
	axes1 = fig1.add_axes([0.1,0.1,0.8,0.8])
	x_axis = np.linspace(-len(corr)/2, len(corr)/2, num=len(corr), endpoint=True, dtype=float)
	axes1.plot(x_axis, corr)
	axes1.set_title('Correlation ' + name)
	axes1.set_xlabel('Sample')
	return axes1

# plot a group of correlations
def plot_correlations(correlations):
	fig1 = plt.figure()
	axes1 = fig1.add_axes([0.1,0.1,0.8,0.8])
	for correlation in correlations:
		corr = correlation[0]
		corr = corr / np.max(corr)
		name = correlation[1]
		x_axis = np.linspace(-len(corr)/2, len(corr)/2, num=len(corr), endpoint=True, dtype=float)
		axes1.plot(x_axis, corr, label=name)
	axes1.set_title('Correlations (normalized to maximum peak height)')
	axes1.legend(loc='lower right')
	axes1.set_xlabel('Sample')
	return axes1

# turn on (or off) group of sources, with center and n sources on either side
# (note that n=0 means that only the center source will be turned on)
def turnon_sources(center, n, sources, on=True):
	for i in range(center-n, center+n+1):
		sources[i].set_on(on=on)

# set all sources to transmit the same signal
def set_all_signals(sources, signal):
	for src in sources:
		src.set_signal(signal)