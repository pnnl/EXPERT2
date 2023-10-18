from colour import Color
from PIL import ImageColor



red = Color("red")
colors = list(red.range_to(Color("green"),10))
text_colors = [color.get_hex() for color in colors]
text_colors_rgba = [list(ImageColor.getcolor(color, "RGBA")) for color in text_colors]
text_colors_rgba.reverse()
text_colors_rgba = [f'rgba({color[0]},{color[1]},{color[2]},{0.25})' for color in text_colors_rgba] + ['rgba(0,0,255,0.15)']
 
 