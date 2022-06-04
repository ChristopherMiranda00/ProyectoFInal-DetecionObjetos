import cv2
"""  -------------------------- Preprocesamiento -------------------------------------- """
def equalizada(image):

    R, G, B = cv2.split(image)
    eq_R = cv2.equalizeHist(R)
    eq_G = cv2.equalizeHist(G)
    eq_B = cv2.equalizeHist(B)
    imagen_equalizada = cv2.merge([eq_R,eq_G,eq_B])
    
    return imagen_equalizada
