# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

few_shot_qa = """Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
A: Sinclair Lewis
Q: Where in England was Dame Judi Dench born?
A: York
Q: In which decade did Billboard magazine first publish and American hit chart?
A: 30s
Q: From which country did Angola achieve independence in 1975?
A: Portugal
Q: Which city does David Soul come from?
A: Chicago
Q: Who won Super Bowl XX?
A: Chicago Bears
Q: Which was the first European country to abolish capital punishment?
A: Norway
Q: In which country did he widespread use of ISDN begin in 1988?
A: Japan
Q: What is Bruce Willis' real first name?
A: Walter
Q: Which William wrote the novel Lord Of The Flies?
A: Golding
Q: Which innovation for the car was developed by Prince Henry of Prussia in 1911?
A: Windshield wipers
Q: How is musician William Lee Conley better known?
A: Big Bill Broonzy
Q: How is Joan Molinsky better known?
A: Joan Rivers
Q: In which branch of the arts is Patricia Neary famous?
A: Ballet
Q: Which country is Europe's largest silk producer?
A: Italy
Q: The VS-300 was a type of what?
A: Helicopter
Q: At which university did Joseph Goebbels become a doctor of philosophy?
A: Heidelberg
Q: Which prince is Queen Elizabeth II's youngest son?
A: Edward
Q: When did the founder of Jehovah's Witnesses say the world would end?
A: 1914
Q: Who found the remains of the Titanic?
A: Robert Ballard
Q: Who was the only Spice Girl not to have a middle name?
A: Posh Spice
Q: What are the international registration letters of a vehicle from Algeria?
A: DZ
Q: How did Jock die in Dallas?
A: Helicopter accident
Q: What star sign is Michael Caine?
A: Pisces
Q: Who wrote the novel Evening Class?
A: Maeve Binchy
Q: Which country does the airline Air Pacific come from?
A: Fiji
Q: In which branch of the arts does Allegra Kent work?
A: Ballet
Q: Who had a 70s No 1 hit with Billy, Don't Be A Hero?
A: Bo Donaldson & The Heywoods
Q: Banting and Best pioneered the use of what?
A: Insulin
Q: Who directed the movie La Dolce Vita?
A: Federico Fellini
Q: Which country does the airline LACSA come from?
A: Costa Rica
Q: Who directed 2001: A Space Odyssey?
A: Stanley Kubrick
Q: Which is the largest of the Japanese Volcano Islands?
A: Iwo Jima
Q: Ezzard Charles was a world champion in which sport?
A: Boxing
Q: Who was the first woman to make a solo flight across the Atlantic?
A: Amelia Earhart
Q: Which port lies between Puget Sound and Lake Washington?
A: Seattle
Q: In which city were Rotary Clubs set up in 1905?
A: Chicago
Q: Who became US Vice President when Spiro Agnew resigned?
A: Gerald Ford
Q: In which decade of the 20th century was Billy Crystal born?
A: 1940s
Q: Which George invented the Kodak roll-film camera?
A: Eastman
Q: Which series had the characters Felix Unger and Oscar Madison?
A: The Odd Couple
Q: Who along with Philips developed the CD in the late 70s?
A: Sony
Q: Where is the multinational Nestle based?
A: Switzerland
Q: Do You Know Where You're Going To? was the theme from which film?
A: Mahogany
Q: 19969 was the Chinese year of which creature?
A: Rat
Q: In the 90s how many points have been awarded for finishing second in a Grand Prix?
A: 6
Q: Stapleton international airport is in which US state?
A: Colorado
Q: What was Kevin Kline's first movie?
A: Sophie's Choice
Q: Which actor had a Doberman Pinscher called Kirk?
A: William Shatner
Q: What day of the week was the Wall Street Crash?
A: Thursday
Q: The US signed a treaty with which country to allow the construction of the Panama Canal?
A: Columbia
Q: What was Prince's last No 1 of the 80s?
A: Batdance
Q: Man In The Mirror first featured on which Michel Jackson album?
A: Bad
Q: Where was the first battle with US involvement in the Korean War?
A: Suwon
Q: On which Caribbean island did Princess Diana spend he first Christmas after her divorce was announced?
A: Barbuda
Q: In which decade was Arnold Schwarzenegger born?
A: 1950s
Q: Which musical featured the song Thank Heaven for Little Girls?
A: Gigi
Q: The Queen Elizabeth liner was destroyed by fire in the 70s in which harbour?
A: Hong Kong
Q: What breed of dog did Columbo own?
A: Basset hound
Q: What was the first movie western called?
A: Kit Carson
Q: Which Oscar-winning actress was born on exactly the same day as actress Lindsay Wagner?
A: Meryl Streep
Q: Which Amendment to the Constitution brought in prohibition in 1920?
A: 18th
Q: Which oil scandal hit the US in 1924?
A: Teapot Dome Scandal
Q: Phil Collins appeared in which Spielberg film with Robin Williams?
A: Hook""".split("\n")

import torch
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, input_length=0, stop_ids=None):
        super().__init__()
        self.stop_ids = stop_ids
        self.input_length = input_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> bool:
        if self.stop_ids is None:
            return False
        output = input_ids[:, self.input_length:]
        has_stop_ids = []
        for stop_id in self.stop_ids:
            has_stop_id = torch.any(output == stop_id, dim=1)
            has_stop_ids.append(has_stop_id)
        has_stop_ids = torch.stack(has_stop_ids, dim=1)
        return (has_stop_ids.any(dim=1).all())
