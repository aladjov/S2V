import json
from encoder_manager import EncoderManager
from configuration import model_config
import tensorflow as tf

FLAGS = tf.flags.FLAGS

class QuickThoughtSentenceEncoder (object):
    DEFAULT_ASSETS_PATH = '/home/vital/python/playgorund/nlp/assets/sentence_vector/'
    DEFAULT_CONFIGURATION = '/home/vital/python/playgorund/tmp/S2V/model_configs/BS400-W300-S1200-Glove-BC-bidir/eval.json'
    def __init__(self):
        self.manager = EncoderManager()
        self.manager.load_model(self.loadConfiguration())

    def loadConfiguration(self):
        with open(QuickThoughtSentenceEncoder.DEFAULT_CONFIGURATION) as json_file:  
            confDictionary = json.load(json_file)
        return model_config(confDictionary,'fixed')

    def encode(self, data):
        return self.manager.encode(data)

if __name__ == "__main__":
    # tf.flags.DEFINE_string("eval_task", "MSRP",
    #                     "Name of the evaluation task to run. Available tasks: "
    #                     "MR, CR, SUBJ, MPQA, SICK, MSRP, TREC.")
    # tf.flags.DEFINE_string("data_dir", None, "Directory containing training data.")
    tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")
    tf.flags.DEFINE_integer("batch_size", 400, "Batch size")
    tf.flags.DEFINE_boolean("use_norm", False, "Normalize sentence embeddings during evaluation")
    tf.flags.DEFINE_integer("sequence_length", 30, "Max sentence length considered")
    tf.flags.DEFINE_string("model_config", QuickThoughtSentenceEncoder.DEFAULT_CONFIGURATION, "Model configuration json")

    tf.app.flags.DEFINE_string('results_path', '/home/vital/python/playgorund/nlp/assets/sentence_vector', """Path to trained models""")
    tf.app.flags.DEFINE_string('Glove_path', '/home/vital/python/playgorund/nlp/assets/words_vector/', """Path to Glove dictionary""")

    text = ['This week, scientists produced the first real image of a black hole, in a galaxy called Messier 87',
        'The image is not a photograph but an image created by the Event Horizon Telescope (EHT) project',
        'Using a network of eight ground-based telescopes across the world, the EHT collected data to produce the image',
        'The black hole itself is unseeable, as it’s impossible for light to escape from it; what we can see is its event horizon',
        'The EHT was also observing a black hole located at the centre of the Milky Way, but was unable to produce an image',
        'While Messier 87 is further away, it was easier to observe, due to its larger size.']
    encoder = QuickThoughtSentenceEncoder()
    encodedText = encoder.encode(text)
    print(encodedText)