import moviepy.editor as mp
clip = mp.VideoFileClip("../Recordings/Corridor/corridor_surveillance.mp4")
clip_resized = clip.resize(height=360)  # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
clip_resized.write_videofile("../Recordings/Corridor/resized_corridor_surveillance.mp4")#, codec='png')
