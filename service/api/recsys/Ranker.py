import pickle

class Ranker:
    def __init__(self, path):
        self.ranker = pickle.load(open(path, "rb"))
    def recommend(self, user_id):
        reco = []
        try:
            recos = self.ranker_preds[
                self.ranker_preds.user_id == user_id].item_id.tolist()[0]
        except IndexError:
            recos = []
        return recos
