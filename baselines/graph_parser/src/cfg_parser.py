from configparser import ConfigParser, NoOptionError


class MyConfigParser(ConfigParser):

    def __init__(self):
        super().__init__()

    def get(self, section, option):
        try:
            #return super().get(section, option)
            return super().get(section, option, raw=True)
        except NoOptionError:
            print(f"option {option} not found in section {section}")
            return None

    def getint(self, section, option):
        try:
            return super().getint(section, option)
        except NoOptionError:
            print(f"option {option} not found in section {section}")
            return None

    def getfloat(self, section, option):
        try:
            return super().getfloat(section, option)
        except NoOptionError:
            print(f"option {option} not found in section {section}")
            return None

    def getboolean(self, section, option):
        try:
            return super().getboolean(section, option)
        except NoOptionError:
            print(f"option {option} not found in section {section}")
            return None

def get_args(fname, args):
    cfg = ConfigParser()
    cfg.read(fname)

    d = {k: v for k,v in vars(args).items()}

    #[data]
    d["train"]                  = cfg.get("data", "train", fallback=None)
    d["val"]                    = cfg.get("data", "val", fallback=None)
    d["predict_file"]           = cfg.get("data", "predict_file", fallback=None)
    d["external"]               = cfg.get("data", "external", fallback=None)
    d["elmo_train"]             = cfg.get("data", "elmo_train", fallback=None)
    d["elmo_dev"]               = cfg.get("data", "elmo_dev", fallback=None)
    d["elmo_test"]              = cfg.get("data", "elmo_test", fallback=None)
    d["load"]                   = cfg.get("data", "load", fallback=None)
    d["target_style"]           = cfg.get("data", "target_style", fallback=None)
    d["other_target_style"]     = cfg.get("data", "other_target_style", fallback=None)
    d["help_style"]             = cfg.get("data", "help_style", fallback=None)
    d["vocab"]                  = cfg.get("data", "vocab", fallback=None)
    #d["neg_extra_style"]        = cfg.get("data", "neg_extra_style", fallback=None)
                       
    #[training]              
    d["batch_size"]             = cfg.getint("training", "batch_size", fallback=None)
    d["epochs"]                 = cfg.getint("training", "epochs", fallback=None)
    d["beta1"]                  = cfg.getfloat("training", "beta1", fallback=None)
    d["beta2"]                  = cfg.getfloat("training", "beta2", fallback=None)
    d["l2"]                     = cfg.getfloat("training", "l2", fallback=None)
                        
    #[network_sizes]       
    d["hidden_lstm"]            = cfg.getint("network_sizes", "hidden_lstm", fallback=None)
    d["hidden_char_lstm"]       = cfg.getint("network_sizes", "hidden_char_lstm", fallback=None)
    d["layers_lstm"]            = cfg.getint("network_sizes", "layers_lstm", fallback=None)
    d["dim_mlp"]                = cfg.getint("network_sizes", "dim_mlp", fallback=None)
    d["dim_embedding"]          = cfg.getint("network_sizes", "dim_embedding", fallback=None)
    d["dim_char_embedding"]     = cfg.getint("network_sizes", "dim_char_embedding", fallback=None)
    d["early_stopping"]         = cfg.getint("network_sizes", "early_stopping", fallback=None)
    d["gcn_layers"]             = cfg.getint("network_sizes", "gcn_layers", fallback=None)
                         
    #[network]            
    d["pos_style"]              = cfg.get("network", "pos_style", fallback=None)
    d["attention"]              = cfg.get("network", "attention", fallback=None)
    d["model_interpolation"]    = cfg.getfloat("network", "model_interpolation", fallback=None)
    d["loss_interpolation"]     = cfg.getfloat("network", "loss_interpolation", fallback=None)
    d["lstm_implementation"]    = cfg.get("network", "lstm_implementation", fallback=None)
    d["char_implementation"]    = cfg.get("network", "char_implementation", fallback=None)
    d["disable_gradient_clip"]  = cfg.get("network", "disable_gradient_clip", fallback=None)
    d["unfactorized"]           = cfg.getboolean("network", "unfactorized", fallback=None)
    d["emb_dropout_type"]       = cfg.get("network", "emb_dropout_type", fallback=None)
    d["bridge"]                 = cfg.get("network", "bridge", fallback=None)
                      
    #[features]      
    d["disable_external"]       = cfg.getboolean("features", "disable_external", fallback=None)
    d["disable_char"]           = cfg.getboolean("features", "disable_char", fallback=None)
    d["disable_lemma"]          = cfg.getboolean("features", "disable_lemma", fallback=None)
    d["disable_pos"]            = cfg.getboolean("features", "disable_pos", fallback=None)
    d["disable_form"]           = cfg.getboolean("features", "disable_form", fallback=None)
    d["use_elmo"]               = cfg.getboolean("features", "use_elmo", fallback=None)
    d["tree"]                   = cfg.getboolean("features", "tree", fallback=None)
                    
    #[dropout]     
    d["dropout_embedding"]      = cfg.getfloat("dropout", "dropout_embedding", fallback=None)
    d["dropout_edge"]           = cfg.getfloat("dropout", "dropout_edge", fallback=None)
    d["dropout_label"]          = cfg.getfloat("dropout", "dropout_label", fallback=None)
    d["dropout_main_recurrent"] = cfg.getfloat("dropout", "dropout_main_recurrent", fallback=None)
    d["dropout_recurrent_char"] = cfg.getfloat("dropout", "dropout_recurrent_char", fallback=None)
    d["dropout_main_ff"]        = cfg.getfloat("dropout", "dropout_main_ff", fallback=None)
    d["dropout_char_ff"]        = cfg.getfloat("dropout", "dropout_char_ff", fallback=None)
    d["dropout_char_linear"]    = cfg.getfloat("dropout", "dropout_char_linear", fallback=None)
                 
    #[other]      
    d["seed"]                   = cfg.getint("other", "seed", fallback=None)
    d["force_cpu"]              = cfg.getboolean("other", "force_cpu", fallback=None)
                
    #[output]  
    d["quiet"]                  = cfg.getboolean("output", "quiet", fallback=None)
    d["save_every"]             = cfg.getboolean("output", "save_every", fallback=None)
    d["disable_val_eval"]       = cfg.getboolean("output", "disable_val_eval", fallback=None)
    d["enable_train_eval"]      = cfg.getboolean("output", "enable_train_eval", fallback=None)
    d["dir"]                    = cfg.get("output", "dir", fallback=None)

    return d

