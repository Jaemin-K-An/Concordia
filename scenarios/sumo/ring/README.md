# SUMO ring-road stability scenario

This is the microscopic fixture for the later phantom-jam prevention experiment. Build and run
it only with a real SUMO installation:

```bash
make sumo-ring-build
make sumo-ring-run
```

`ring.net.xml` and all outputs are generated, not committed. The route makes five laps so
oscillations have time to form. Vehicle parameters vary desired speed and reaction noise only;
the navigation policy does not control acceleration or lane changes. SSM output is configured
for TTC, PET, and DRAC. A valid phantom-jam conclusion additionally requires the wave detector
and repeated matched seeds; this fixture alone is not evidence.
