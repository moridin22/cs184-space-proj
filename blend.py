from PIL import Image

im1 = Image.open("tests/1.png")
im2 = Image.open("tests/2.png")
im4 = im1.crop((160, 80, 480, 420))
im5 = im2.paste(im4, (160, 80, 480, 420))
f = open("tests/3.png", "w+")
im3 = Image.blend(im1, im2, 100)
im2.save("tests/3.png")
