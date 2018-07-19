import cv2
import tensorflow as tf
import numpy as np

class TLClassifierTest(object):
    def __init__(self):
        #load classifier
        graph_path = './models/model_real/frozen_inference_graph.pb'
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #implement light color prediction
        img_expand = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num_detections) = \
            self.sess.run([self.boxes, self.scores, self.classes, self.num_detections], feed_dict={self.image_tensor:img_expand})
        #boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        num_det = np.squeeze(num_detections).astype(np.int32)
        print('SCORES: ', scores[0])
        print('CLASSES: ', classes[0])
        print('# DETS: ', boxes.shape)
        

        if scores[0] > 0.5:
            if classes[0] == 1:
                print('GREEN')
            elif classes[0] == 2:
                print('RED')
            elif classes[0] == 3:
                print('YELLOW')
        else:
            print('NONE')


    def load_graph(self, frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")
        return graph

if __name__ == "__main__":
    classifier = TLClassifierTest();
    image_path = '' #MODIFY THIS LINE TO GIVE THE CLASSIFIER AN IMAGE
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    classifier.get_classification(image);
    
