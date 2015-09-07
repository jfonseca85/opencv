'''
Created on 05/09/2015

@author: Jorge Luis
Abrir uma imagem colorida, visualizar e salvar.
'''
from skimage import io
import sys
import dlib

class LoadImage(object): 
    def __init__(self,pathImage):
        '''
            Construtor do LoadIamge
        '''
        self.pathImage = pathImage
        self.win = dlib.image_window()
        
    def printImage(self):
        img = io.imread(self.pathImage)
        self.win.set(img)
        self.win.add_overlay(img)
        #self.win.se
        
if __name__ == '__main__':
    image = LoadImage('C://opencv//sources//data//images//guil.jpg')
    win = dlib.image_window()
    #win.set_title(('image_window')
    win.set_image(image)
    dlib.hit_enter_to_continue()
