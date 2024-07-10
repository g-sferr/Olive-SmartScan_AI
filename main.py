
import torch
import torch.nn as nn
import torchvision.models as models
from src.models.detection_model import test

def main():
    #Code for test functions of the module
    ''' 
    data_dir = 'path/to/data'
    dataloader = load_data(data_dir)

    detector = OliveDetector()
    counter = OliveCounter()

    detector.load_state_dict(torch.load('detector.pth'))
    counter.load_state_dict(torch.load('counter.pth'))

    detector.eval()
    counter.eval()

    for inputs, _ in dataloader:
        with torch.no_grad():
            detections = detector(inputs)
            bounding_boxes, confidences = detector.decode_outputs(detections)
            high_conf_boxes = bounding_boxes[confidences > 0.5]
            
            # Normalizza le coordinate delle bounding boxes
            image_size = inputs.size(2)  # assuming square images
            high_conf_boxes = high_conf_boxes / image_size

            # Usa roi_align per ritagliare le immagini
            crops = ops.roi_align(inputs, [high_conf_boxes], output_size=(32, 32))
            
            for crop in crops:
                count = counter(crop)
                print(f'Count: {count.item()}')

            visualize_predictions(inputs[0], high_conf_boxes, None)
            '''
    test()


if __name__ == '__main__':
    # Launch Application
    main()