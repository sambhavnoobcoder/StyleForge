import gradio as gr
import torch
from transformers import logging
import random
from PIL import Image
from Utils import MingleModel

logging.set_verbosity_error()


def get_concat_h(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    dst = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
      dst.paste(im, (x_offset,0))
      x_offset += im.size[0]
    return dst


mingle_model = MingleModel()


def mingle_prompts(first_prompt, second_prompt):
    imgs = []
    text_input1 = mingle_model.do_tokenizer(first_prompt)
    text_input2 = mingle_model.do_tokenizer(second_prompt)
    with torch.no_grad():
        text_embeddings1 = mingle_model.get_text_encoder(text_input1)
        text_embeddings2 = mingle_model.get_text_encoder(text_input2)

    rand_generator = random.randint(1, 2048)
    # Mix them together
    # mix_factors = [0.1, 0.3, 0.5, 0.7, 0.9]
    mix_factors = [0.5]
    for mix_factor in mix_factors:
        mixed_embeddings = (text_embeddings1 * mix_factor + text_embeddings2 * (1 - mix_factor))

        # Generate!
        steps = 20
        guidence_scale = 8.0
        img = mingle_model.generate_with_embs(mixed_embeddings, rand_generator, num_inference_steps=steps,
                                 guidance_scale=guidence_scale)
        imgs.append(img)

    return get_concat_h(imgs)


with gr.Blocks() as demo:
    gr.Markdown(
        '''
        <h1 style="text-align: center;"> Fashion Generator GAN</h1>
        ''')

    gr.Markdown(
        '''
        <h3 style="text-align: center;"> Note : the gan is extremely resource extensive, so it running the inference on cpu takes long time . kindly wait patiently while the model generates the output. </h3>
        ''')
    
    gr.Markdown(
        '''
        <p style="text-align: center;">generated an image as an average of 2 prompts inserted !!</p>
        ''')

    first_prompt = gr.Textbox(label="first_prompt")
    second_prompt = gr.Textbox(label="second_prompt")
    greet_btn = gr.Button("Submit")
    # gr.Markdown("## Text Examples")
    # gr.Examples([['batman, dynamic lighting, photorealistic fantasy concept art, trending on art station, stunning visuals, terrifying, creative, cinematic',
    #               'venom, dynamic lighting, photorealistic fantasy concept art, trending on art station, stunning visuals, terrifying, creative, cinematic'],
    #              ['A mouse', 'A leopard']], [first_prompt, second_prompt])

    gr.Markdown("# Output Results")
    output = gr.Image(shape=(512,512))

    greet_btn.click(fn=mingle_prompts, inputs=[first_prompt, second_prompt], outputs=[output])

demo.launch()

