# Invoke AI Tools

This is a collection of scripts for use with [Invoke AI](https://github.com/invoke-ai/InvokeAI).

## General Installation

- These are intended to be ran under the same virtual environment that InvokeAI runs under.
- Even still, there's probably a couple packages you'll need to `pip install` manually (such as `rich`). This is left as an exercise to the reader.

## invoke-png

This tool is a bit of a swiss army knife for dumping the metadata from Invoke-created PNG files.

## invoke-train

This script is used to create a new Textual Inversion embedding.

This is a naive approach to training a model for use with InvokeAI. It's not perfect, but it works.

It was written as a direct port of an existing notebook that demonstrated the technique. This is simply a convenient wrapper around that. I'm not well versed enough in the underlying technology to know if this is the best way to do it. There may be much better, much more optimized ways to do it. (PRs welcome.)

### Quick usage:

1. `invoke-train init project_name`
2. Fill out the form (defaults are usually fine). YMMV.
3. A directory will be created with the "project_name".
4. Inside this, toss your training images into the `images` directory.
   1. Images should be pre-processed to fit a 1:1 square aspect ratio at 512x512.
      - _The only reason this is not done automatically is because the focal point of a cropped 1:1 image is up to you. I can't guess that. Unless you're doing faces, maybe that's possible. But this is not restricted to faces, so... get croppin'._
5. `invoke-train train project_name` to begin the training. (This takes _about_ a half-hour on a 3090 at 3000 steps.)
6. When it's done, `invoke-train install project_name` will copy the newly created embed into your InvokeAI installation root as dictated by the `INVOKEAI_ROOT` environment variable.

## Trivia

- A good chunk of the argument handling and overall structure in both projects, so far, was made quickly thanks to AI assisted coding using CoPilot. It killed a lot of the repetition and made it easier to focus on what features I wanted to see, instead of dicking around looking up documentation.

-- Fortyseven (Network47.org), 2023-04-16
