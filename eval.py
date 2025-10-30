# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
"""Evaluation script for Nerf."""
import functools
from os import path

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np
import tensorflow as tf
import optax
import torch
import lpips as lpips_torch

from jaxnerf.nerf import datasets
from jaxnerf.nerf import models
from jaxnerf.nerf import utils

FLAGS = flags.FLAGS
utils.define_flags()


def compute_lpips(image1, image2, model):
  """Compute the LPIPS metric using PyTorch implementation."""
  image1 = np.asarray(image1, dtype=np.float32)
  image2 = np.asarray(image2, dtype=np.float32)

  # Ensure both are RGB
  if image1.ndim == 2:
    image1 = np.repeat(image1[..., None], 3, axis=-1)
  if image2.ndim == 2:
    image2 = np.repeat(image2[..., None], 3, axis=-1)
  if image1.shape[-1] == 1:
    image1 = np.repeat(image1, 3, axis=-1)
  if image2.shape[-1] == 1:
    image2 = np.repeat(image2, 3, axis=-1)

  t1 = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
  t2 = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
  t1, t2 = t1.cpu(), t2.cpu()

  with torch.no_grad():
    dist = model(t1, t2)
  return float(dist.item())


def main(unused_argv):
  # Prevent TensorFlow from grabbing GPUs
  tf.config.experimental.set_visible_devices([], "GPU")
  tf.config.experimental.set_visible_devices([], "TPU")

  rng = random.PRNGKey(20200823)

  if FLAGS.config is not None:
    utils.update_flags(FLAGS)
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set.")

  dataset = datasets.get_dataset("test", FLAGS)
  rng, key = random.split(rng)
  model, init_variables = models.get_model(key, dataset.peek(), FLAGS)

  tx = optax.adam(FLAGS.lr_init)
  state = utils.TrainState.create(
      apply_fn=model.apply,
      params=init_variables["params"],
      tx=tx,
  )
  del init_variables

  print("Loading LPIPS (VGG) from PyTorch...")
  lpips_model = lpips_torch.LPIPS(net="vgg").cpu()
  lpips_model.eval()

  # Deterministic rendering
  def render_fn(variables, key_0, key_1, rays):
    return jax.lax.all_gather(
        model.apply(variables, key_0, key_1, rays, False), axis_name="batch"
    )

  render_pfn = jax.pmap(
      render_fn,
      in_axes=(None, None, None, 0),
      donate_argnums=3,
      axis_name="batch",
  )

  ssim_fn = jax.jit(functools.partial(utils.compute_ssim, max_val=1.), backend="cpu")

  last_step = 0
  out_dir = path.join(
      FLAGS.train_dir, "path_renders" if FLAGS.render_path else "test_preds"
  )
  if not FLAGS.eval_once:
    summary_writer = tensorboard.SummaryWriter(path.join(FLAGS.train_dir, "eval"))

  while True:
    # Restore checkpoint into new-style TrainState
    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    step = int(state.step)  # <-- fixed

    if step <= last_step:
      continue

    if FLAGS.save_output and not utils.isdir(out_dir):
      utils.makedirs(out_dir)

    psnr_values, ssim_values, lpips_values = [], [], []

    if not FLAGS.eval_once:
      showcase_index = np.random.randint(0, dataset.size)

    for idx in range(dataset.size):
      print(f"Evaluating {idx+1}/{dataset.size}")
      batch = next(dataset)

      pred_color, pred_disp, pred_acc = utils.render_image(
          functools.partial(render_pfn, state.params),  # <-- fixed
          batch["rays"],
          rng,
          FLAGS.dataset == "llff",
          chunk=FLAGS.chunk,
      )

      if jax.host_id() != 0:
        continue

      if not FLAGS.eval_once and idx == showcase_index:
        showcase_color, showcase_disp, showcase_acc = pred_color, pred_disp, pred_acc
        if not FLAGS.render_path:
          showcase_gt = batch["pixels"]

      if not FLAGS.render_path:
        psnr = utils.compute_psnr(((pred_color - batch["pixels"]) ** 2).mean())
        ssim = ssim_fn(pred_color, batch["pixels"])
        lpips_val = compute_lpips(pred_color, batch["pixels"], lpips_model)
        print(f"PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips_val:.4f}")
        psnr_values.append(float(psnr))
        ssim_values.append(float(ssim))
        lpips_values.append(float(lpips_val))

      if FLAGS.save_output:
        utils.save_img(pred_color, path.join(out_dir, f"{idx:03d}.png"))
        utils.save_img(pred_disp[..., 0], path.join(out_dir, f"disp_{idx:03d}.png"))

    if not FLAGS.eval_once and jax.host_id() == 0:
      summary_writer.image("pred_color", showcase_color, step)
      summary_writer.image("pred_disp", showcase_disp, step)
      summary_writer.image("pred_acc", showcase_acc, step)
      if not FLAGS.render_path:
        summary_writer.scalar("psnr", np.mean(psnr_values), step)
        summary_writer.scalar("ssim", np.mean(ssim_values), step)
        summary_writer.scalar("lpips", np.mean(lpips_values), step)
        summary_writer.image("target", showcase_gt, step)

    if FLAGS.save_output and not FLAGS.render_path and jax.host_id() == 0:
      np.savetxt(path.join(out_dir, f"psnrs_{step}.txt"), psnr_values)
      np.savetxt(path.join(out_dir, f"ssims_{step}.txt"), ssim_values)
      np.savetxt(path.join(out_dir, f"lpips_{step}.txt"), lpips_values)
      np.savetxt(path.join(out_dir, "psnr.txt"), [np.mean(psnr_values)])
      np.savetxt(path.join(out_dir, "ssim.txt"), [np.mean(ssim_values)])
      np.savetxt(path.join(out_dir, "lpips.txt"), [np.mean(lpips_values)])

    if FLAGS.eval_once or step >= FLAGS.max_steps:
      break

    last_step = step


if __name__ == "__main__":
  app.run(main)
