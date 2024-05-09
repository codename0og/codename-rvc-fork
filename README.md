## Essentially an 'up-to-date RVC' <br />  <br />✨ Enhanced with everything I find useful, practical and neat ✨ <br />  <br />ㅤㅤㅤFork creator:   Codename;0 <br />ㅤ
### **If you need to contact me, here's my discord:ㅤ.codename0.**
> tho, please, let me know right away you come from github.
> Also, keep in mind I don't keep people on my friend list for too long,<br />so don't get surprised with the removal after we sort our things out lol - no hate, that's how I am.
<br />

#### ㅤ <br />Important info:
> 1. This fork is using mainline RVC ( repo, not releases ) as a base.
> 2. Crucial point of this fork is to be up-to-date with the mainline.
> 3. Some features and ui elements ( overal ui style ) borrowed from mangio's fork. - I kinda dislike the new one lol.
> 4. I might port / borrow few things from other forks 'cause why not. If there are thigs **you** want to see added in here, let me know.
⠀<br />
⠀<br />
## ⚠️ PYTHON COMPATIBILITY WARNING ⚠️
### The fork is confirmed and tested-in-field to work with python 3.10.6 with my requirements list. <br />( DO NOT USE python 3.12.0, it won't work. )


<br />


# Fork's features:
**| Training related |**

- Mangio's crepe F0/Feature extraction method
It is the best ( Not the fastest tho. ) extraction method for models that rely on clean ( No reverb, delay, harmonies, noise etc. ) or truly HQ datasets.
- Adjustable hop_length for Mangio's crepe method ( Definitely the biggest perk of using Mangio-crepe. )
- Envelopes for processed samples / segments to avoid zero-crossing ( waveform interruption ) clicks.
- My own " Mel similarity metric " as a bonus. Helps in spotting overtraining / overfitting and mode collapses. Metric is being displayed in the console / log and is also logged in tensorboard files.
- Sox is used for resampling ( vhq - Very High quality as default )

⠀<br />
**| Inference related |**

- Automatic index file matching for models.
- Auto detection of audio files in " audios " folder + a dropdown to choose 'em.
- Changed default Feature index search ratio to '0.5'.
- Mangio-crepe ( standard; " full " ) - Go to method.
- FCPE f0/pitch estimation method added - tweaked and ported from applio by me.

⠀<br />
**| ONNX Export related |**
- There's been many issues and problems with newest releases' onnx exporting so I decided to port 0618's ( RVC 0618 v2 beta release ) way of handling it.
( tl;dr, now exporting works. It exports the onnx models with onnx 14's operators.  ~ Tested having onnx 15 runtime - so the latest one available. )

I've confirmed the exported .onnx models to work with:
- my "RVC_Onnx_Infer" thingy: https://github.com/codename0og/RVC_Onnx_Infer ( and generally rvc's own original onnx inference demo ) 
- As for w-okada, nope. Shapes mismatch.
( I suppose that's pretty normal and in order to use w-okada with onnx, you gotta convert the rvc's .pth to .onnx in the w-okada itself. )

⠀<br />
## Bugs / Stuff that's broken.

- Noticed that after you unload the model from inference the " protect voiceless consonants " value becomes null.
FIX: After you unload the voice / model and then load any again, simply move the slider or input the default value: 0.33
( I've re-checked the code again during the big update and.. honestly? idk, code seems fine and handling too so idk why it happens lol.
It's either something that's always been there and I didn't notice or mangio-style ui breaks somethin' I dunno. whatever. )

- No other bugs or issues I've observed. In case you do and it's serious / critical, write me a msg on discord.

⠀<br />
## Potentially to-do ( might be abandoned if I change my mind lel ):
- onnx inference within the web-ui ( separate tab ) + have it use rvc's slicing / caching mechanism⠀<br />(( I have it in wip state in older dev fork build, the ui and onnx model loading in works, but it's all just a dummy input. No inference yet
- voice changer within the web-ui
- re-organize all the files ( structure re-interpretation
- change / tweak the style of the interface
- model similarity comparitor ( haven't gotten into it yet )
- automated 'accurate' log interval syncing

⠀<br />
## CONCEPT STAGE - Stuff I kinda have in mind but am unsure if I wanna go that way.
- Native ONNX inference
I kinda worked on it but mehh, due to the reasons I've written about in my onnx infer thingy I sorta don't feel a need to work on it, at least for now.

- new additional slicing mechanism.. perhaps whisper or maybe other ml assisted approach ( would have a checkbox )
- incorporating yt-dlp and ffmpeg conversion for yt-sourced audio.. hmm.. maybe even all in 1 'simple' dataset preparator lol, maybe.
- in-built tts solution
- auto dl'er for voice models ( dunno of that yet )
- dataset evaluation mechanism ( fidelity scoring )
<br />


# Installation and usage guide:
### Everything is easy and automated. Just read carefully and follow please:
<br />

- Step 1. Download the "codename-rvc-fork-v1.0.0.zip" from the release section.
<br />

- Step 2. Unpack the content of the .zip ( 'codename-rvc-fork-v1.0.0' folder ) to your preferred location.

#### ⚠️ IMPORTANT: ㅤThe path to the unpacked rvc folder cannot contain spaces or funky symbols.
#### An example of a good path:
> D:\AI_related\voice-cloning\codename-rvc-fork-v1.0.0
<br />
ㅤ
ㅤ

- Step 3. In the fork's folder, go to:ㅤㅤ'requirements/RVC' and pick the right .bat for you:

#### For standard Nvidia users:
> ' install requirements for RVC ( NVIDIA-STANDARD ).bat '

#### For Nvidia RTX30xx users:
> ' install requirements for RVC ( NVIDIA-RTX30XX ).bat '

#### IF THE ABOVE ↑ CAUSES YOUR GPU TO BE UNDETECTED / OTHER GPU RELATED PROBLEMS, <br />USE THE FOLLOWING ONE:
> ' install requirements for RVC ( NVIDIA-RTX30XX )- CUDA_118.bat '
<br />

ㅤ
ㅤ
- Step 4. Go back to the main rvc folder and use this to download required assets:
> ' download-required-assets.bat '
<br />

- Step 5. Run the RVC with:
> ' RUN_WEB-UI.bat '
<br />


### Where to put downloaded RVC voice models?
> Put the '.pth' model files in:ㅤㅤ'assets/weights'
<br />


> Put the index files:ㅤㅤ'logs/folder_with_the_same_name_as_the_pth_model'
<br />

### END OF THE GUIDE
<br />

# Here are my other thingies if you're interested:
- Standalone and lightweight RVC's native real-time-voice-changer:
> https://github.com/codename0og/rvc-realtime-voice-changer
- Inferencing for RVC's onnx models:
> https://github.com/codename0og/RVC_Onnx_Infer
