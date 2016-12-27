# organize-yaself

Here's a little repo for testing out some ideas about self organizing maps I saw in the <a href="http://library.seg.org/series/segeab">SEG Abstracts</a>:
<ul>
<li><a href="http://library.seg.org/doi/abs/10.1190/segam2016-13949728.1">Zhao, et al. "Advanced self-organizing map facies analysis with stratigraphic constraint"</a>
<li><a href="http://library.seg.org/doi/abs/10.1190/segam2015-5924540.1">Zhao, et al. "Supervised and unsupervised learning: how machines can assist quantitative seismic interpretation"</a>
</ul>

<a href="https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/">Here's</a> the awesome article by Sachin Joglekar about building SOMs in <a href="https://tensorflow.org">Tensorflow</a>. His work is reproduced here with his permission in several of my scripts. Enjoy.

Another color clustering <a href="https://github.com/spiglerg/Kohonen_SOM_Tensorflow">example</a> by spiglerg. This one trains on GPU.

Notes on the plan going forward (20161227):
<ul>
<li>Model reflection response based on rock properties at various angles. The vector of angles will be used as an input "prototype" feature vector. This set represents the training set.
<li><li>There could be multiple attributes for each feature vector, including phase, freq, etc.
<li>The output space will have the same dimensionality as the lithologic characteristics input to the waveform modeling algorithm, e.g. shaliness, porosity, bulk modulus, grain size, etc.
<li>The real seismic (prestack) will be input to the SOM after training.
<li>The output classifications can be thresholded or colormapped for any property(ies) you wish to examine.
<li>The modeling algo's the key. The color SOM in this repo can be adapted with an nD output.
<li>Do I need shot gathers or image gathers for classification (assuming only PP response)? Image; though it'll lack perfect amp response, the energy mispositioning pre-mig will certainly screw up the classification.
</ul>
