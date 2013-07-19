import moose
import matplotlib.pylab as pl

n = moose.Neutral('/n')
c = moose.Compartment('/n/c')
dt = 1.0

h = moose.HSolveCuda('/n/h')
h.dt = dt
h.target = '/n/c'

t = moose.Table('/n/t')
table_conn = moose.connect(t, 'requestData', c, 'get_Vm')

p = moose.PulseGen('/n/p')
p.delay[0] = 2
p.width[0] = 100
p.level[0] = 0.100
p.delay[1] = 100
pulse_conn = moose.connect(p, 'outputOut', c, 'injectMsg')

pt = moose.Table('/n/pt')
pt_conn = moose.connect(pt, 'requestData', p, 'get_output')

moose.setClock(0, dt)
moose.useClock(0, '/n/h', 'process')
moose.useClock(0, '/n/t', 'process')
moose.useClock(0, '/n/p', 'process')
moose.useClock(0, '/n/pt', 'process')

moose.reinit()
moose.start(10)

pl.subplot(211)
pl.plot(range(len(t.vec)), t.vec)
pl.subplot(212)
pl.plot(range(len(pt.vec)), pt.vec)
pl.show()
