#!/usr/bin/env python3
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import repeat
from scipy import misc
from scipy.io import wavfile
from scipy.ndimage import filters
from numpy import fft
from multiprocessing import Pool

def make_parser():
    parser = argparse.ArgumentParser(description='Convolve image with sound')
    parser.add_argument('image', help='Input Image')
    parser.add_argument('wav', help='Input Wav')
    parser.add_argument('--fps', help='Frame rate', default=60, type=int)
    parser.add_argument('--output', help='Save output as video file')
    parser.add_argument('--sample-count-factor', default=1, type=float, help='For each frame, after taking into account '\
        'sampling rate of the sound, the number of sound samples to sampled will be multiplied by this factor')
    return parser

def update_frame(generated_image, height, width):
    plt.imshow(generated_image)


def make_image_generator(image, wav, sample_rate, fps, sample_count_factor):
    interval = 1/fps
    interval_samples = round(interval*sample_rate*sample_count_factor)
    sample_count = len(wav)

    height = len(image)
    width = len(image[0])

    reds = image[:, :, 0]
    greens = image[:, :, 1]
    blues = image[:, :, 2]

    channels = [reds, greens, blues]

    def generator():
        index = 0
        with Pool(3) as p:
            while index < sample_count:
                start = time.clock()

                samples = wav[index:min(index + interval_samples, sample_count)]
                index += interval_samples

                convolution_matrix = abs(fft.rfft2(samples, norm='ortho'))
                normalized = convolution_matrix/np.linalg.norm(convolution_matrix)

                (convolved_red, convolved_green, convolved_blue) = p.starmap(convolve_channel,
                                                                             zip(channels, repeat(normalized)))

                convolved = np.zeros((height, width, 3), 'uint8')
                convolved[:, :, 0] = convolved_red
                convolved[:, :, 1] = convolved_green
                convolved[:, :, 2] = convolved_blue

                end = time.clock()
                time_taken = end - start
                print("Frame took %ss FPS met: %s" % (time_taken, time_taken < interval))

                yield convolved
            print("Done")

    return generator


def convolve_channel(channel, convolution_matrix):
    return filters.convolve(channel, convolution_matrix)

def main():
    parser = make_parser()
    args = parser.parse_args()

    image = misc.imread(args.image, mode='RGB')

    height = len(image)
    width = len(image[0])

    (sample_rate, wav) = wavfile.read(args.wav)
    sample_count = len(wav)
    duration = sample_count/sample_rate

    interval = 1000/args.fps

    fig = plt.figure()
    ani = animation.FuncAnimation(fig, update_frame,
            make_image_generator(image, wav, sample_rate, args.fps, args.sample_count_factor),
            fargs=(height, width),
            interval=interval)
    if args.output is not None:
        Writer = animation.writers['ffmpeg']
        writer = Writer()
        print("Saving to", args.output)
        ani.save(args.output, writer=writer)

    else:
        plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    main()
