# General

Right now you need to to re-run the whole pipeline when switching between reward function presets (conservative or aggressive), because the model files get overwritten (so backup them).

If args for a script are unclear, run the script with -h to get list of arguments

Run tensorboard: `python3 -m tensorboard.main --logdir runs/ --host 0.0.0.0`

Tensorboard runs are in runs directory

# Train state based policy

Without noise augmentation:

`python3 sac_continous_action.py`

With noise augmentation:

`python3 sac_continous_action.py --noise-augment`

# Fill replay buffer

Trajectories are driven by state based policy

Default (clean model):

`python3 fillRB.py`

Using the noise augmented state policy:

`python3 fillRB.py --noisy`

# Train autoencoder

`python vae.py`

# Train vision based policy using BC

`python3 bc.py --model-type vision`

`python3 bc.py --model-type vision --noisy`

# Train vision based policy using DAgger (using state actor as expert)

Default (clean state expert):

`python3 dagger.py`

Using the regular noise augmented state expert:

`python3 dagger.py --noisy`

# Evaluate models

- Clean: `python3 eval.py --model-type state`
- Noise augmented: `python3 eval.py --model-type state --noisy`

Evaluate a DAgger-trained vision model (uses the replay-buffer basename logic):

- DAgger (vision): `python3 eval.py --model-type vision --dagger`
- DAgger (vision, noise augmented): `python3 eval.py --model-type vision --dagger --noisy`

Evaluation report file is create in eval directory

# Create video

`python3 actorInference.py`