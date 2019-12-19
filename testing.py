import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig
import sigsim as s

# all unitless distances assumed to be meters throughout project

duration = 10.0 # sec
samplerate = 1000
v_signal = 2.5 # velocity (m/s) of signal travel from src to rec

N_sources = 36
radius = 10.0


##### Generate signal #####
times = np.linspace(0, duration, num=np.floor(samplerate*duration).astype(int), endpoint=True, dtype=float)

sig = np.sinc(4*times - duration)
for i in range(int(samplerate*duration/2), len(sig)):
	sig[i] = 0


##### Set up sources, receivers #####
sources = s.create_sources(radius, N_sources, on=False)
receivers = [ s.Receiver('A',-1,0), s.Receiver('B',1,0), s.Receiver('C',0,1) ]

s.set_all_signals(sources, sig)
center = 0
s.turnon_sources(center, 0, sources)

s.plot_map(sources, receivers)


##### Collect data #####
s.record_data(sources, receivers, v_signal, samplerate, eq_time=True)

s.plot_receiver_signal(times, receivers[0])
s.plot_receiver_signal(times, receivers[1])


##### Analyze data #####
## Demonstrate correlation-convolution property in three-receiver setup ##
corrAB = s.correlate(receivers[0], receivers[1])
s.plot_correlation(corrAB).set_xlim(-6000,6000)

corrBC = s.correlate(receivers[1], receivers[2])
s.plot_correlation(corrBC).set_xlim(-6000,6000)

corrAC = s.correlate(receivers[0], receivers[2])
s.plot_correlation(corrAC).set_xlim(-6000,6000)

corrAC_calc_data = np.convolve(corrAB[0], corrBC[0])
corrAC_calc = (corrAC_calc_data, 'AB-BC convolution')

s.plot_correlations([corrAC, corrAC_calc]).set_xlim(-6000,6000)


## Compare performance with different sources enabled ##
maxs_AC = np.empty(N_sources)
maxs_AC_calc = np.empty(N_sources)

for i in range(N_sources):
	s.clear_receiver_data(receivers)
	sources[i-1].set_off()
	sources[i].set_on()

	s.record_data(sources, receivers, v_signal, samplerate, eq_time=True)

	corrAB = s.correlate(receivers[0], receivers[1])
	corrBC = s.correlate(receivers[1], receivers[2])
	corrAC = s.correlate(receivers[0], receivers[2])
	maxs_AC[i] = np.argmax(corrAC[0]) - len(corrAC[0])/2

	corrAC_calc_data = np.convolve(corrAB[0], corrBC[0])
	corrAC_calc = (corrAC_calc_data, 'AB-BC convolution')
	maxs_AC_calc[i] = np.argmax(corrAC_calc[0]) - len(corrAC_calc[0])/2

fig1 = plt.figure()
axes1 = fig1.add_axes([0.1,0.1,0.8,0.8])
x_axis = 360.0/N_sources * np.arange(N_sources)
axes1.plot(x_axis, maxs_AC, 'ro', label='AC cross-correlation')
axes1.plot(x_axis, maxs_AC_calc, 'b.', label='AB-BC convolution')
axes1.set_title('Center Offset of Cross-Correlation for AC Receiver Pair')
axes1.set_xlabel('Degrees around circle')
fig1.legend(loc='lower right')

plt.show()
