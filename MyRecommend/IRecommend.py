from abc import abstractmethod, ABC


class IRecommend(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def recommend(self, user):
        pass
