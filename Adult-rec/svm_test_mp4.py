import pylab
#import imageio
filename = 'AA-AN-DL.mp4'
#vid = imageio.get_reader(filename,  'ffmpeg')
#print(type(vid))
#nums = [10, 287]
#for num in nums:
#    image = vid.get_data(num)
#    fig = pylab.figure()
#    fig.suptitle('image #{}'.format(num), fontsize=20)
#    pylab.imshow(image)
#pylab.show()

from moviepy.editor import *
 
def mp4tomp3(mp4file,mp3file):
    videoclip=VideoFileClip(mp4file)
    audioclip=videoclip.audio
    audioclip.write_audiofile(mp3file)
    audioclip.close()
    videoclip.close()
 
#mp4tomp3(filename,"audio.mp3")
mp3 = read_mp3(mp3_filename)