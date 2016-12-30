The modeler in this directory builds wavelets by stacking out a single trace after modeling with the Aki-Richards approximation to the Zoeppritz equations. The A-R form used here can be found in Mavko et al., 2009, p.101:

<img src="https://latex.codecogs.com/gif.latex?R(\theta&space;)&space;=&space;\frac{1}{2}\left(&space;\frac{\Delta&space;V_{p}}{V_{p}}&space;&plus;\frac{\Delta&space;\rho&space;}{\rho&space;}\right&space;)&plus;\left&space;[&space;\frac{1}{2}&space;\frac{\Delta&space;V_{p}}{V_{p}}&space;-&space;2k&space;\left&space;(&space;2&space;\frac{\Delta&space;V_{s}}{V_{s}}&space;&plus;&space;\frac{\Delta&space;\rho&space;}{\rho&space;}&space;\right&space;)\right&space;]sin^{2}(\theta&space;)&plus;\frac{1}{2}&space;\frac{\Delta&space;V_{p}}{V_{p}}&space;(tan^{2}(\theta&space;)-sin^{2}(\theta&space;))" title="R(\theta ) = \frac{1}{2}\left( \frac{\Delta V_{p}}{V_{p}} +\frac{\Delta \rho }{\rho }\right )+\left [ \frac{1}{2} \frac{\Delta V_{p}}{V_{p}} - 2k \left ( 2 \frac{\Delta V_{s}}{V_{s}} + \frac{\Delta \rho }{\rho } \right )\right ]sin^{2}(\theta )+\frac{1}{2} \frac{\Delta V_{p}}{V_{p}} (tan^{2}(\theta )-sin^{2}(\theta ))" />

I've included the stacking operation because the pre-stack version of the SOM is going to classify across an event in offset space, so I'll need the full operator.


BTW these equations were built with the awesome <a href="https://www.codecogs.com/latex/eqneditor.php">CodeCogs</a>.
