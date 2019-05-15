import os
import imageio
import subprocess
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def make_gif(FROM, TO, num_frames, framerate, sys_args, filename):
    step_size = (np.array(TO) - np.array(FROM)) / (int(num_frames) - 1)
    frames = []
    for i in range(0, num_frames):
        frames.append('tests/gifs/' + str(i) + '.png')
        position = np.array(FROM) + (step_size * int(i))
        position = [float(str.strip(str(x))) for x in position]
        program_arg = "-g" + str(i) + ";" + str(position[0]) + ";" + str(position[1]) + ";" + str(position[2])
        subprocess.call(["python", "tracer.py", program_arg, str(sys_args[0]), sys_args[1]], stderr=open(os.devnull, 'w'))
    images = []
    for i in range(0, num_frames):
        images.append(imageio.imread(frames[i]))
        os.remove(frames[i])
    imageio.mimsave('tests/gifs/' + filename + '.gif', images, duration = float(framerate))
