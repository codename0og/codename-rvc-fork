# Fork creator:   Codename;0
**You can contact on discord:    .codename0.**


## Codename's fork features:
| Training related |

- Mangio's crepe F0/Feature extraction method
It is the best ( Not the fastest tho. ) extraction method for models that rely on clean ( No reverb, delay, harmonies, noise etc. ) or truly HQ datasets.
- Adjustable hop_length for Mangio's crepe method ( More accurate f0 / feature extraction. Can capture more granual and subtle details )
- Envelopes for processed samples / segments to avoid clicks resulting from slicer's ignorance of zero-crossing.
- " Mel similarity metric ". A lil thingy I've made to help myself during my rvc-learning stage in spotting overtraining / overfitting and mode collapses. Metric is being displayed in the console / log and is also logged in tensorboard files.

Explanation on what "hop length " is and what is it's purpose for mangio-crepe in rvc:

A custom hop length is like choosing between seeing tiny details or getting a bigger picture in a photo.
Zooming In: Smaller hop lengths let you see small changes in sound, like zooming in for more detail in a picture. It's detailed but takes more time to process.
Zooming Out: Larger hop lengths give a wider view of the sound, like stepping back from a picture. It's quicker for the computer but shows fewer small changes.
It's about picking how closely you want to examine the sound: up close for more details (but slower) or from a distance to see more overall (but faster).
Adjusting this helps match what you need to know with how much time you have for the analysis.


| Inference related |

- Automatic index file matching for models.
- Auto detection of audio files in " audios " folder + a dropdown to choose 'em. ( Much easier than having to copy/paste paths to your .wav or whatever all the time )
- Changed default Feature index search ratio to '0.5'.
- Mangio-crepe ( standard; " full " ) - Go to method.
NOTES: Speed is definitely not it's strength ( like it's the case in RMVPE ) and it's also way way more sensitive to dirty audio, however if you provide it clean audio, magic happens. 
**( As always, for best possible results I recommend doing inferences on both RMVPE and Mangio-Crepe and just mixing the results during a song / cover mixing. )**

- Mangio-crepe-tiny ( " lite " version of mangio-crepe; " tiny " ) - Better than pm, harvest but worse than crepe, mangio crepe and most likely on par with rmvpe ( not as robust tho, that's for sure ).
- FCPE - An f0 extraction method adapted from rvc's real-time-vc ( Borrowed from applio, re-tweaked for my fork. ) Speed comparable to RMVPE, Quality as well. FCPE is a context based extractor so, def works the best in real-time. Added it in cause why not.


| ONNX Export related |
- There's been many issues and problems with newest releases' onnx exporting so I decided to port 0618's ( RVC 0618 v2 beta release ) way of handling it.
( tl;dr, now exporting works. It exports the onnx models with onnx 14's operators.  ~ Tested having onnx 15 runtime - so the latest one available. )

I've confirmed the exported .onnx models to work with;
- my "RVC_Onnx_Infer" thingy: https://github.com/codename0og/RVC_Onnx_Infer ( and generally rvc's own original onnx inference demo ) 
- As for w-okada, nope. Shapes mismatch.
( I suppose that's pretty normal and in order to use w-okada with onnx, you gotta convert the rvc's .pth to .onnx in the w-okada itself. ) 



## RMVPE or Mangio-Crepe?
You see, each has own strenghts and weaknesses and foremostly, each serves a different purpose in a way.
Mangio-crepe is a "Crepe" modified by Mangio so we'll refer to it as Crepe in here.

-------------------------------------------------------------------------------------------------------------------------

Crepe is a mono-phonic type of extraction/estimation, meaning;
- it doesn't handle well audio which contains: layers, harmonies or any other difficulties in estimating the correct / dominant f0 path.
( results in pitch breaks, snaps, gurgly sounds etc. )


Pros:
- It tends to make the output a lil softer and smoother
( Imo it's a plus and tends to improve realism and overal smoothness + I feel like it stays more true to datasets compared to rmvpe )
- adjustable hop allows you to capture way more details than rmvpe can allow you to.
( default ish hop for mangio training and inference is 64. Rmvpe for training sits at 160 and inference is done at 512, so it's not as detail-rish as you can hopefully understand. )
- Much better handling of whispery-type of content, breathing, sighs, moans etc. Generally, " noise " type of audio data.

Cons:
- It's way more sensitive than harvest or rmvpe; It can mistake the artifacts in audio ( such as post cleanup / post uvr: instrumental residues and such alike )
- It's sound:noise detection capabilities are worse than RMVPE so, if there's too much of noise, humm etc., the extraction performance will suffer.
- Quite heavy on gpu ( vRAM )
- May cause memory leaks ( or so Mangio himself mentioned in his og repo. )
- Sometimes the model's convergence ( during training ) might be slower or not as good as it is with rmvpe.

-------------------------------------------------------------------------------------------------------------------------

RMVPE is a polyphonic type of extraction/estimation, meaning;
- It does handle well audio which contains: layers, harmonies or audio-elements that'd be difficult to interpret by other f0/feature methods.
( Of course, there are obvious limitations. RMVPE was made with intention of being a robust method, rather than completely replacing vocal-instru separation.)

Pros:
- It's way way lighter on GPU ( esp. vram department wise ) compared to crepe / mangio-crepe
- It's speed is no joke. Hella lightning fast.
- It tends to make the vocal / speach sharper? more uhm, perhaps, vivid/solid? It's hard to explain unless you compare the rmvpe with mangio yourself.
( same as Mangio's softness. It can be seen as a plus or minus, depending on a user and use-case. ) 

Cons:
- It can make breathing and such "noise" type of sounds, harsher - but it ain't a rule!.
- Sometimes ( not always! ) rmvpe-trained models have a metalic overlay-tone to it and using rmvpe on such models, further enhances the effect.
- I'd say, it's not as " soft " as crepe-based methods but you know. It's a matter of use-case. Sometimes you want that, sometimes not.
- You can't adjust the hop length whatsoever. RMVPE has it set to 160 or so ~ 


-------------------------------------------------------------------------------------------------------------------------

## Bugs / Stuff that's broken.

- Noticed that after you unload the model from inference the " protect voiceless consonants " value becomes null.
FIX: After you unload the voice / model and then load any again, simply move the slider or input the default value: 0.33
( I've re-checked the code again during the big update and.. honestly? idk, code seems fine and handling too so idk why it happens lol. It's either something that's always been there and I didn't notice or mangio-style ui breaks somethin' I dunno. whatever. )

- No other bugs or issues I've captured. In case you do and it's serious, write me a msg on discord.

