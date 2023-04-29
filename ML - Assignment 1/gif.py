import sys
import datetime
import imageio
import os

VALID_EXTENSIONS = ('png', 'jpg')



def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(output_file, images, duration=duration)


if __name__ == "__main__":
    # script = sys.argv.pop(0)
    #
    # if len(sys.argv) < 2:
    #     print('Usage: python {} <duration> <path to images separated by space>'.format(script))
    #     sys.exit(1)
    #
    # duration = float(sys.argv.pop(0))
    # filenames = sys.argv
    # filenames = []
    path = "D:\\Studies\\ML\\ML - Assignment 1 - Datasets\\images\\"
    files = os.listdir(path)
    files.sort()
    print(files)

    filenames = [path + i for i in files]
    # print(len(filenames))
    duration = 0.1



    if not all(f.lower().endswith(VALID_EXTENSIONS) for f in filenames):
        print('Only png and jpg files allowed')
        sys.exit(1)

    create_gif(filenames, duration)