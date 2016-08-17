# Glitcher

```
usage: glitcher.py [-h] [--fps FPS] [--output OUTPUT]
                   [--sample-count-factor SAMPLE_COUNT_FACTOR]
                   image wav

Convolve image with sound

positional arguments:
  image                 Input Image
  wav                   Input Wav

optional arguments:
  -h, --help            show this help message and exit
  --fps FPS             Frame rate
  --output OUTPUT       Save output as video file
  --sample-count-factor SAMPLE_COUNT_FACTOR
                        For each frame, after taking into account sampling
                        rate of the sound, the number of sound samples to
                        sampled will be multiplied by this factor
```
