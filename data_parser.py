import csv 

class SFData:
    def __init__(self, file_name):
        self.matches = {}

        with open(file_name, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')

            # trackers as we parse
            last_round_index = -1
            last_round_winner = -1
            rounds_won0 = 0
            rounds_won1 = 0

            for row in reader:
                match_name = row['Match_No']

                round_index = int(row['Match_Round'][-1]) - 1

                # create new if doesn't exist
                if match_name not in self.matches:
                    self.matches[match_name] = {}

                    self.matches[match_name]['match_winner'] = -1
                    self.matches[match_name]['round_winners'] = []

                    self.matches[match_name]['series'] = []

                    self.matches[match_name]['max_healths'] = (float(row['Player_1_MaxHealth']), float(row['Player_2_MaxHealth']))

                    # reset trackers
                    last_round_index = -1
                    last_round_winner = -1
                    rounds_won0 = 0
                    rounds_won1 = 0

                self.matches[match_name]['series'].append((float(row['Player_1']), float(row['Player_2']), rounds_won0, rounds_won1))

                if self.matches[match_name]['match_winner'] == -1 and row['Game_Winner'] != '0':
                    self.matches[match_name]['match_winner'] = int(row['Game_Winner'] == 'Player 2')

                round_winner = int(row['Winner'] == 'Player 2')

                # if new round
                if last_round_index != round_index:
                    self.matches[match_name]['round_winners'].append(round_winner)

                    if last_round_winner != -1:
                        if last_round_winner == 0:
                            rounds_won0 += 1
                        else:
                            rounds_won1 += 1

                last_round_index = round_index
                last_round_winner = round_winner

if __name__ == '__main__':
    data = SFData('data.csv')

    for match_name, match_data in data.matches.items():
        print(match_name)
        print(match_data)
        break
