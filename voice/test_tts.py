import torch
import time
from kokoro import KPipeline, KModel
import soundfile as sf
import os

def speed_callable(len_ps):
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1

voice_dir = '../ckpts/kokoro-v1.1/voices/'
voice_files = [f for f in os.listdir(voice_dir) if f.startswith('zf') and f.endswith('.pt')]

# 创建输出文件夹
output_dir = 'output_voices'
os.makedirs(output_dir, exist_ok=True)

repo_id = 'hexgrad/Kokoro-82M-v1.1-zh'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '../ckpts/kokoro-v1.1/kokoro-v1_1-zh.pth'
config_path = '../ckpts/kokoro-v1.1/config.json'
model = KModel(model=model_path, config=config_path, repo_id=repo_id).to(device).eval()

for voice_file in voice_files:
    zh_pipeline = KPipeline(lang_code='z', repo_id=repo_id, model=model)
    sentence = '广东拓斯达科技股份有限公司成立于2007年，2017年上市，全球研发总部基地坐落于松山湖。拓斯达坚持“让工业制造更美好”的企业使命，秉承“成为全球领先的智能装备服务商”的企业愿景，通过以机器人、注塑机、CNC为核心的智能装备，以及控制、伺服、视觉三大核心技术，打造以人工智能驱动的智能硬件平台，为制造企业提供智能工厂整体解决方案。'
    voice_name = os.path.splitext(voice_file)[0]
    print(f"Testing voice: {voice_name}")
    voice_zf_tensor = torch.load(os.path.join(voice_dir, voice_file), weights_only=True)
    start_time = time.time()
    generator = zh_pipeline(sentence, voice=voice_zf_tensor, speed=speed_callable)
    result = next(generator)
    wav = result.audio
    speech_len = len(wav) / 24000
    print('yield speech len {}, rtf {}, time {}'.format(speech_len, (time.time() - start_time) / speech_len, time.time() - start_time )) 
    sf.write(os.path.join(output_dir, f'output_{voice_name}.wav'), wav, 24000)



# # 3️⃣ Initalize a pipeline
# from kokoro import KPipeline
# from IPython.display import display, Audio
# import soundfile as sf
# import torch
# import time
# # 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# # 🇪🇸 'e' => Spanish es
# # 🇫🇷 'f' => French fr-fr
# # 🇮🇳 'h' => Hindi hi
# # 🇮🇹 'i' => Italian it
# # 🇯🇵 'j' => Japanese: pip install misaki[ja]
# # 🇧🇷 'p' => Brazilian Portuguese pt-br
# # 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
# pipeline = KPipeline(lang_code='z',device='cuda') # <= make sure lang_code matches voice, reference above.

# # This text is for demonstration purposes only, unseen during training
# # text = '''
# # The sky above the port was the color of television, tuned to a dead channel.
# # "It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. "It's like my body's developed this massive drug deficiency."
# # It was a Sprawl voice and a Sprawl joke. The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese.

# # These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come. One of the main issues was the shape that Christian churches were to take, since there was not, apparently, a tradition of monumental church buildings when Constantine decided to help the Christian church build a series of truly spectacular structures. The main form that these churches took was that of the basilica, a multipurpose rectangular structure, based ultimately on the earlier Greek stoa, which could be found in most of the great cities of the empire. Christianity, unlike classical polytheism, needed a large interior space for the celebration of its religious services, and the basilica aptly filled that need. We naturally do not know the degree to which the emperor was involved in the design of new churches, but it is tempting to connect this with the secular basilica that Constantine completed in the Roman forum (the so-called Basilica of Maxentius) and the one he probably built in Trier, in connection with his residence in the city at a time when he was still caesar.

# # [Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
# # '''
# # text = '「もしおれがただ偶然、そしてこうしようというつもりでなくここに立っているのなら、ちょっとばかり絶望するところだな」と、そんなことが彼の頭に思い浮かんだ。'
# text = '不好意思，饮料不够了'
# # text = 'Los partidos políticos tradicionales compiten con los populismos y los movimientos asamblearios.'
# # text = 'Le dromadaire resplendissant déambulait tranquillement dans les méandres en mastiquant de petites feuilles vernissées.'
# # text = 'ट्रांसपोर्टरों की हड़ताल लगातार पांचवें दिन जारी, दिसंबर से इलेक्ट्रॉनिक टोल कलेक्शनल सिस्टम'
# # text = "Allora cominciava l'insonnia, o un dormiveglia peggiore dell'insonnia, che talvolta assumeva i caratteri dell'incubo."
# # text = 'Elabora relatórios de acompanhamento cronológico para as diferentes unidades do Departamento que propõem contratos.'

# # 4️⃣ Generate, display, and save audio files in a loop.
# time_start = time.time()
# generator = pipeline(
#     text, voice='zf_xiaoxiao', # <= change voice here
#     speed=1.0, split_pattern=r'\n+'
# )
# # zf_xiaoxiao
# # Alternatively, load voice tensor directly:
# # voice_tensor = torch.load('path/to/voice.pt', weights_only=True)
# # generator = pipeline(
# #     text, voice=voice_tensor,
# #     speed=1, split_pattern=r'\n+'
# # )

# for i, (gs, ps, audio) in enumerate(generator):
#     print(i)  # i => index
#     print(gs) # gs => graphemes/text
#     print(ps) # ps => phonemes
#     # display(Audio(data=audio, rate=24000, autoplay=i==0))
#     sf.write(f'{i}.mp3', audio, 24000) # save each audio file
# time_end = time.time()
# print(f"Time taken: {time_end - time_start} seconds")

