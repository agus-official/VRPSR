import codecsimulator
from PIL import Image
import numpy as np
import time

img = Image.open("src.png")
qp = 35

# H.264
codec = codecsimulator.PYRGBx264Codec(512, 512)
img_np = np.asarray(img)
out_np = np.zeros_like(img_np)

start_time = time.time()
sz = codec.Encode(img_np, out_np, qp)
elapsed_time = time.time() - start_time

out = Image.fromarray(out_np, 'RGB')
out.save("rec_264_qp_{}_rate_{}_time_{:.3f}s.png".format(qp, sz, elapsed_time))


# H.265
codec = codecsimulator.PYRGBx265Codec(512, 512)
img_np = np.asarray(img)
out_np = np.zeros_like(img_np)

start_time = time.time()
sz = codec.Encode(img_np, out_np, qp)
elapsed_time = time.time() - start_time

out = Image.fromarray(out_np, 'RGB')
out.save("rec_265_qp_{}_rate_{}_time_{:.3f}s.png".format(qp, sz, elapsed_time))


# H.266
codec = codecsimulator.PYRGBx266Codec(512, 512)
img_np = np.asarray(img)
out_np = np.zeros_like(img_np)

start_time = time.time()
sz = codec.Encode(img_np, out_np, qp)
elapsed_time = time.time() - start_time

out = Image.fromarray(out_np, 'RGB')
out.save("rec_266_qp_{}_rate_{}_time_{:.3f}s.png".format(qp, sz, elapsed_time))
