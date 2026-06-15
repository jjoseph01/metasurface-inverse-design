# Inverse design of metasurfaces for controlling thermal radiation

Designing a metasurface usually goes one slow direction: you draw a geometry, run a heavy
electromagnetic simulation, look at the absorbance it produces, tweak the shape, and run the
simulation again. Hours later you might land on something close to what you wanted. This project
runs that loop backwards. You tell the model the absorbance curve you want, and it hands you
geometries that should produce it — in a couple of seconds, no simulator in the loop at inference
time.

We built a conditional Wasserstein GAN with a gradient penalty (cWGAN-GP) and trained it alongside
a CNN "simulator" that learned to predict a geometry's absorbance from its image. During training
the simulator sits behind the generator and grades it on physics, not just on whether the shapes
look real. That's the part that makes the generated designs actually mean something.

This was a 2025 NSF REU project at the University of North Texas. Advised by Dmytro Shymkiv and Dr. Yuzhe Xiao.

![Project poster](figures/poster.png)

## How it fits together

Three networks, trained in two stages.

**The simulator** (`RobustSimulatorCNN` in `models.py`) is a residual CNN that takes a 64×64
grayscale geometry and predicts its 15-point absorbance curve (one value per incidence angle, from
0 to 14·π/30). We train it first, on its own, until it's a fast and accurate stand-in for the real
electromagnetic simulation. Then we freeze it.

**The generator** takes a latent noise vector plus the absorbance curve you're asking for, and
upsamples that into a 64×64 geometry through a stack of transposed convolutions.

**The critic** is the WGAN-GP discriminator. It learns to tell real dataset geometries from
generated ones, which keeps the generator producing shapes that look physically plausible instead
of noise.

The generator's loss has two pieces: the adversarial term from the critic (look real) and a
simulator term (the frozen simulator's predicted absorbance for the generated shape should match
the target curve). The second term is what ties the picture to the physics. The simulator loss
(`ShapeAndMagnitudeLoss`) isn't a plain MSE either — it's an L1 on the values plus an L1 on the
curve's gradient, so the model is rewarded for getting the *shape* of the absorbance curve right,
not just the average height.

## Running it

You'll need PyTorch (install it first, matched to your CUDA version, from the PyTorch site), then:

```bash
pip install -r requirements.txt
```

Drop your data into a `Data/` folder next to the code, laid out like this:

```
Data/
  Data_Generated_Images/        # the geometry images (subfolders per shape class are fine)
  metasurface_absorbance_compiled_final.csv   # filename + 15 Absorbance columns per row
  target_responses.csv          # the absorbance curves you want the GAN to hit (one per row)
```

Then:

```bash
python main.py
```

Everything you'd want to change lives in `config.py` — training mode, image size, batch size,
learning rates, how hard the simulator pushes the generator (`LAMBDA_SIM_LOSS`), and so on. The two
modes:

- `CONSTANT_TARGET` trains a fresh generator for one target curve at a time. Good for nailing a
  specific design.
- `CONDITIONAL` trains a single generator you can later hand any target curve to.

Set `PRETRAINED_SIMULATOR_PATH` to a saved checkpoint to skip simulator pre-training, or to `None`
to train it from scratch.

Each run writes to its own timestamped folder under `output_*/`: simulator loss curves, GAN
loss/learning-rate curves, the generated geometries, and DTW-ranked comparison plots of predicted
vs. target absorbance for every target.

## What's in each file

| file | what it does |
|------|--------------|
| `config.py` | every hyperparameter and path |
| `data_loader.py` | datasets + stratified train/test split |
| `models.py` | generator, critic, and the residual-CNN simulator |
| `losses_optimizers.py` | the shape+magnitude loss, optimizers, schedulers, gradient penalty |
| `training_loops.py` | simulator pre-training and both GAN training loops |
| `graph_output.py` | all the evaluation plots (uses DTW to rank generated designs) |
| `utils.py` | weight initialization |
| `main.py` | wires it all together and runs the full pipeline |

## Honest notes

- The generator works at 64×64. It's enough to capture the shapes, but the rounder, blobbier
  geometries pick up jagged edges that probably cost a little absorbance accuracy.
- Absorbance is sampled at only 15 angles, so very sharp spectral features are hard to reproduce.
- The whole thing is only as accurate as the simulator it trains against — a better simulator would
  raise the ceiling on everything downstream.
- Our dataset is thin on geometries whose absorbance peaks between 5·π/30 and 10·π/30, which shows
  up as a gap when you look at the design space.

## Acknowledgements

Supported by Dr. Ting Xiao's NSF REU "Beyond Language: Training to Create and Share Vector
Embeddings across Applications" (award #2244259). Thanks to Dr. Ting Xiao, Dr. Mark Albert, and
Haoxuan Zhang for running a great program, and to Dmytro Shymkiv and Dr. Yuzhe Xiao for the
guidance.
