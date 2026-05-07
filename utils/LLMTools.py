#利用MLLM生成更详细的描述词
import torch
import re




template_fg_phrase_q='Provide a concise and comprehensive descriptive compound noun phrase for {}, without any environmental description, using only noun phrases and prepositional phrases, emphasizing its shape, color, and texture.'
template_bg_phrase_q='Provide a concise and comprehensive descriptive compound noun phrase for the environment that is highly similar to {}, using only noun phrases and prepositional phrases.'
template_fg_word_q='Name of the {} in one word.'
template_bg_word_q='Name of the environment of the {} in one word.'
prompt_qkeys_dict={

    'TheCamo':          ['camouflaged animal'],
    'TheShadow':        ['shadow'],
    'TheGlass':         ['glass'],
    'ThePolyp':         ['polyp'],

    '3attriTheBgSyn':   ['concealed animal', 'hidden animal', 'unseen animal'],
    '3attriTheBgSynCamo':   ['camouflaged object', 'camouflaged animal', 'camouflaged entity'],
    # '3attriTheBgSynCamo':   ['camouflaged animal'],
    '3attriTheBgSynCamoSpec':   ['camouflaged species', 'disguised species', 'hidden species'],

    '3TheGlassSyn':     ['glass', 'window', 'mirror'],
    '3TheGlassSyn1':     ['glass', 'window', 'transparent material'],

    '3TheShadowSyn':    ['shadow', 'silhouette', 'profile'],
    '3TheShadowSyn1':    ['shadow', 'silhouette', 'outline'],

    '3ThePolypSyn':     ['polyp', 'appendage', 'tentacle'],
    '3ThePolypSyn1':    ['polyp', 'appendage', 'tumor'],
    '3ThePolypSyn2':    ['polyp', 'tumor', 'growth'],

    '1attriTheCamouflageBg_test': ['camouflaged animal'],
    '3attriTheBgSynCamo_test':   ['camouflaged animal', 'disguised animal', 'hidden animal'],

}
prompt_q_dict={}
for k, v in prompt_qkeys_dict.items():
    if prompt_q_dict.get(k) is None:
        prompt_q_dict[k] = [[template_fg_phrase_q.format(key), template_bg_phrase_q.format(key), template_fg_word_q.format(key),template_bg_word_q.format(key)] for key in prompt_qkeys_dict[k]]
prompt_gene_dict={}
for k, v in prompt_qkeys_dict.items():
    if prompt_gene_dict.get(k) is None:
        prompt_gene_dict[k] = [prompt_qkeys_dict[k], ['environment']]


def expand_bbox(bbox, expansion_rate=0.15):
    if not bbox or len(bbox) != 4:
        return bbox

    x1, y1, x2, y2 = bbox

    original_width = x2 - x1
    original_height = y2 - y1

    expand_width = original_width * expansion_rate
    expand_height = original_height * expansion_rate

    new_x1 = max(0, x1 - expand_width / 2)
    new_y1 = max(0, y1 - expand_height / 2)
    new_x2 = min(1, x2 + expand_width / 2)
    new_y2 = min(1, y2 + expand_height / 2)

    return [new_x1, new_y1, new_x2, new_y2]

def get_text_from_img_llava(
    pil_img, prompt_q,
    model, image_processor, tokenizer,
    get_bg_text=False,
    image_tensor=None,
    conv_mode='llava_v0',
    temperature=0.2,
    w_caption=False,
    target_class_name=None,
    device=None):
    '''
    input
    '''
    # from transformers import TextStreamer
    from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from LLaVA.llava.conversation import conv_templates, SeparatorStyle
    # from llava.model.builder import load_pretrained_model
    from LLaVA.llava.utils import disable_torch_init
    from LLaVA.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

    # prompt_qkeys_l = prompt_qkeys_dict[prompt_q]
    # question_l = prompt_q_dict[prompt_q]

    if target_class_name and isinstance(target_class_name, str) and target_class_name.strip(): 
        target = target_class_name.strip()
        prompt_qkeys_l = [target]
        question_l = [[
            template_fg_phrase_q.format(target),
            template_bg_phrase_q.format(target),
            template_fg_word_q.format(target),
            template_bg_word_q.format(target)
        ]]
    else:
        prompt_qkeys_l = prompt_qkeys_dict[prompt_q]
        question_l = prompt_q_dict[prompt_q]
        
    textfg_phrase_list = []
    textbg_phrase_list = []
    textfg_word_list = []
    textbg_word_list = []
    bbox_ori_list = []

    image = pil_img #load_image(img_path)
    image_width, image_height = image.size
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(device)
    # llm_dtype = next(model.parameters()).dtype
    # image_tensor = image_tensor.to(dtype=llm_dtype)

    disable_torch_init()
    for qi, qs in enumerate(question_l):

        if w_caption:
            q_keyword = prompt_qkeys_l[qi]
            bbox_naive = [0, 0, 1, 1]
            caption_q = f'This image is from {q_keyword} segmentation task, describe the {q_keyword} in one sentence'
            bbox_q = f' The naive bounding box of {q_keyword} is {bbox_naive}, adjust the bounding box to ensure that all {q_keyword} are fully displayed. Just output the adjusted bounding box.'
            qs=[caption_q] + qs + [bbox_q]

        image = pil_img #load_image(img_path)
        conv = conv_templates[conv_mode].copy() # 是否需要改一下system 提示词，换成caption？

        for i, inp in enumerate(qs):

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) #不让再终端打印生成的描述

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=1024,
                    # streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().lower()
            conv.messages[-1][-1] = outputs

            if w_caption and i == 0:
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '')  # "<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
            if w_caption and i in [1,3]:
                if outputs.find('"') > 0:
                    outputs = outputs.split('"')[1]
                elif outputs.find(' is an ') > 0:
                    outputs = outputs.split(' is an ')[1]
                elif outputs.find(' is a ') > 0:
                    outputs = outputs.split(' is a ')[1]
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '')  # "<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                if outputs.find('.'):
                    outputs = outputs.split('.')[0]

                if outputs.find(' is '):
                    outputs = outputs.split(' is ')[0]
                if outputs.find(' are '):
                    outputs = outputs.split(' are ')[0]
                if outputs.find('that'):
                    outputs = outputs.split('that')[0]
                if outputs.find(' in which '):
                    outputs = outputs.split(' in which ')[0]
                if outputs.find(' which '):
                    outputs = outputs.split(' which ')[0]
                if outputs[-1] == ',':
                    outputs = outputs[:-1]


                while outputs[0] == ' ':  outputs = outputs[1:]
                while outputs[-1] == ' ':  outputs = outputs[:-1]



                if i == 1:
                    textfg_phrase_list.append(outputs)
                else:
                    textfg_word_list.append(outputs)

            if w_caption and i in [2,4]:
                if outputs.find('"') > 0:
                    outputs = outputs.split('"')[1]
                elif outputs.find(' is an ') > 0:
                    outputs = outputs.split(' is an ')[1]
                elif outputs.find(' is a ') > 0:
                    outputs = outputs.split(' is a ')[1]
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '')  # "<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                if outputs.find('.'):
                    outputs = outputs.split('.')[0]

                if outputs.find(' is '):
                    outputs = outputs.split(' is ')[0]
                if outputs.find(' are '):
                    outputs = outputs.split(' are ')[0]
                if outputs.find('that'):
                    outputs = outputs.split('that')[0]
                if outputs.find(' in which '):
                    outputs = outputs.split(' in which ')[0]
                if outputs.find(' which '):
                    outputs = outputs.split(' which ')[0]
                if outputs[-1] == ',':
                    outputs = outputs[:-1]

                while outputs[0] == ' ':  outputs = outputs[1:]
                while outputs[-1] == ' ':  outputs = outputs[:-1]

                if i == 2:
                    if outputs == textfg_phrase_list[-1]:
                        textbg_phrase_list.append('background')
                    else:
                        textbg_phrase_list.append(outputs)
                else:
                    if outputs == textfg_word_list[-1]:
                        textbg_word_list.append('background')
                    else:
                        textbg_word_list.append(outputs)
            if w_caption and i == 5:

                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '')  # "<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                outputs = outputs.strip('[]</s> \n')
                string_numbers = re.findall(r'\d+\.\d+', outputs)
                outputs_bbox = [round(float(num), 2) for num in string_numbers]
                outputs_bbox = expand_bbox(outputs_bbox, 0.0)
                bbox_ori = [0, 0, 1, 1]
                if not outputs_bbox or len(outputs_bbox) != 4:
                    bbox_ori[0], bbox_ori[1], bbox_ori[2], bbox_ori[3] = int(bbox_ori[0] * image_width), int(bbox_ori[1] * image_height), int(bbox_ori[2] * image_width), int(bbox_ori[3] * image_height)
                    bbox_ori_list.append(bbox_ori)
                elif (outputs_bbox[2] - outputs_bbox[0]) * (outputs_bbox[3] - outputs_bbox[1]) == 0:
                    bbox_ori[0], bbox_ori[1], bbox_ori[2], bbox_ori[3] = int(bbox_ori[0] * image_width), int(bbox_ori[1] * image_height), int(bbox_ori[2] * image_width), int(bbox_ori[3] * image_height)
                    bbox_ori_list.append(bbox_ori)
                if len(outputs_bbox) >= 4:
                    bbox_ori[0], bbox_ori[1], bbox_ori[2], bbox_ori[3] = int(outputs_bbox[0] * image_width), int(outputs_bbox[
                        1] * image_height), int(outputs_bbox[2] * image_width), int(outputs_bbox[3] * image_height)
                    bbox_ori_list.append(bbox_ori)


    # if len(textbg_phrase_list)==0:
    #     textbg_phrase_list=['background']
    # if len(textfg_phrase_list)==0:
    #     textfg_phrase_list=['hidden animal']


    return textfg_phrase_list, textbg_phrase_list,textfg_word_list, textbg_word_list,bbox_ori_list