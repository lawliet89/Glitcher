#!/usr/bin/env python3
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import misc
from scipy.io import wavfile
from scipy.ndimage import filters
from numpy import fft

SOUND_THROTTLE_CONSTANT = 100

def make_parser():
    parser = argparse.ArgumentParser(description='Convolve image with sound')
    parser.add_argument('image', help='Input Image')
    parser.add_argument('wav', help='Input Wav')
    parser.add_argument('--fps', help='Frame rate', default=60, type=int)
    return parser

def update_frame(generated_image, height, width):
    plt.imshow(generated_image)


def make_image_generator(image, wav, sample_rate, fps):
    interval = 1/fps
    interval_samples = round(interval*sample_rate/SOUND_THROTTLE_CONSTANT)
    sample_count = len(wav)

    pixels = [pixel for row in image for pixel in row]
    image_fft = fft.rfft(pixels, norm="ortho")
    height = len(image)
    width = len(image[0])

    def generator():
        index = 0
        while index < sample_count:
            start = time.clock()

            samples = wav[index:min(index + interval_samples, sample_count)]
            index += interval_samples

            # convolution_matrix = abs(fft.rfft2(samples, norm='ortho'))
            # normalized = convolution_matrix/np.linalg.norm(convolution_matrix)
            # convolved = filters.convolve(image, normalized)

            samples_mono = samples.sum(axis=1)/2

            sound_fft = fft.rfft(samples_mono, norm="ortho")
            convolved_freq = np.convolve(image_fft, sound_fft, "same")

            convolved = fft.irfft(convolved_freq, norm="ortho")
            convolved = np.reshape(convolved, (height, width))

            end = time.clock()
            print("Frame took %s" % (end - start))

            yield convolved

    return generator


def main():
    parser = make_parser()
    args = parser.parse_args()

    image = misc.imread(args.image, True)
    height = len(image)
    width = len(image[0])

    (sample_rate, wav) = wavfile.read(args.wav)
    sample_count = len(wav)
    duration = sample_count/sample_rate

    interval = 1000/args.fps

    fig = plt.figure()
    ani = animation.FuncAnimation(fig, update_frame, make_image_generator(image, wav, sample_rate, args.fps),
            fargs=(height, width),
            interval=interval, repeat=False)
    plt.show()

    # import matplotlib.pyplot as plt
    # plt.imshow(f)
    # plt.show()

    # pixels = [pixel for row in image for pixel in row]
    # mono_wav = wav.sum(axis=1)/2

    # image_fft = fft.rfft(pixels, norm="ortho")
    # sound_fft = fft.rfft(mono_wav, 1024, norm="ortho")
    #
    # convolved = np.convolve(image_fft, sound_fft, "same")
    # convolved_image = fft.irfft(convolved, norm="ortho")

    # convolution_matrix = abs(fft.rfft2(wav[args.offset:(args.offset + args.samples)], norm='ortho'))
    # normalized = convolution_matrix/np.linalg.norm(convolution_matrix)
    # convolved = filters.convolve(image, normalized)
    # misc.imsave(args.output, convolved)

if __name__ == "__main__":
    # execute only if run as a script
    main()
