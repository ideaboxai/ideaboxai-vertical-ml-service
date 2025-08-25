from ml_services.scoring_ranking.strategies.base import BaseScoringRankingStrategy


class TOPSISRankerStrategy(BaseScoringRankingStrategy):
    """Concrete implementation of the TOPSIS ranking strategy."""

    def score_and_rank(self, dataset, nCol, impact, weights=None):
        normalized_data = self._normalize_data(dataset.copy(), nCol, weights)
        ideal_best, ideal_worst = self._calculate_ideal_values(
            normalized_data, nCol, impact
        )
        separation_best, separation_worst = self._calculate_separation_measures(
            normalized_data, ideal_best, ideal_worst, nCol
        )
        relative_closeness = self._calculate_relative_closeness(
            separation_best, separation_worst
        )
        normalized_data["relative_closeness"] = relative_closeness
        normalized_data["rank"] = normalized_data["relative_closeness"].rank(
            ascending=False
        )

        return normalized_data.sort_values(by="rank")

    def _normalize_data(self, dataset, nCol, weights=None):
        for i in range(1, nCol):
            temp = 0

            for j in range(len(dataset)):
                temp += dataset.iloc[j, i] ** 2
            temp = temp**0.5

            for j in range(len(dataset)):
                if weights is not None:
                    dataset.iloc[j, i] = (dataset.iloc[j, i] / temp) * weights[i - 1]
                else:
                    dataset.iloc[j, i] = dataset.iloc[j, i] / temp

        return dataset

    def _calculate_ideal_values(self, dataset, nCol, impact):
        ideal_best = []
        ideal_worst = []

        for i in range(1, nCol):
            if impact[i - 1] == "+":
                ideal_best.append(dataset.iloc[:, i].max())
                ideal_worst.append(dataset.iloc[:, i].min())
            else:
                ideal_best.append(dataset.iloc[:, i].min())
                ideal_worst.append(dataset.iloc[:, i].max())

        return ideal_best, ideal_worst

    def _calculate_separation_measures(self, dataset, ideal_best, ideal_worst, nCol):
        separation_best = []
        separation_worst = []

        for i in range(len(dataset)):
            best = 0
            worst = 0

            for j in range(1, nCol):
                best += (dataset.iloc[i, j] - ideal_best[j - 1]) ** 2
                worst += (dataset.iloc[i, j] - ideal_worst[j - 1]) ** 2

            separation_best.append(best**0.5)
            separation_worst.append(worst**0.5)

        return separation_best, separation_worst

    def _calculate_relative_closeness(self, separation_best, separation_worst):
        relative_closeness = []

        for i in range(len(separation_best)):
            if separation_best[i] + separation_worst[i] == 0:
                relative_closeness.append(0)
            else:
                relative_closeness.append(
                    separation_worst[i] / (separation_best[i] + separation_worst[i])
                )

        return relative_closeness
