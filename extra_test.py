# from diffusers import DiffusionPipeline
# from diffusers.utils import load_image
#
# # pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", cache_dir='/scratch/b502b586')
#
# prompt = "Turn this cat into a dog"
# input_image = load_image(
# "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
#
#
# # image = pipe(image=input_image, prompt=prompt).images[0]
#
#
#
# from diffusers import DiffusionPipeline
# from diffusers.utils import load_image
#
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", cache_dir='/scratch/b502b586')
# #
# pipe.to("cuda")
# prompt = "Turn this cat into a dog"
# input_image = load_image(
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
#
#
# image = pipe(image=input_image, prompt=prompt, num_inference_steps=100,).images[0]
#
# plt.figure()
# plt.imshow(input_image)
# plt.savefig('orig.png')
#
# plt.figure()
# plt.imshow(image)
# plt.savefig('ss.png')