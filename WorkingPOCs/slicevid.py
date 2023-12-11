pf=r'C:\Users\logan\Desktop\MEng\Mask_RCNN\IceData\SplashVideos\GoodCopySplahofColor.avi'

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip(pf, 0, 60, targetname="GoodCopySplahofColor_trimmed.avi")