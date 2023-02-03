---
layout: post
title:  "Magic: The Gathering Calculator"
date:   2023-02-02 13:27:23 -0800
categories: jekyll update
---
# Project Description

In *Magic: the Gathering*, deck building is often both tricky and
irrational. Mostly, players use gut feel to guide their decisions. In a
game as complex as Magic, this is often the only recourse and honestly
is part of the draw for a lot of players.

In this paper, we will be looking to make deck building a little more
rational with the help of math. Specifically, we will be looking to
optimize the card costs so our opening hand of seven cards is as good as
possible. We will do this by manually collecting data points to classify
hands as "good" or "not good." From here, we use these data points to
train a neural network to classify hands for us. Finally, using this
neural network, we can apply a Multi-armed Bandits algorithm to
different deck configurations to determine which configuration has the
highest change of giving us a good hand.

# Context and *Magic: The Gathering* Basics

Before we dive into the math, we must first provide some additional
context. The cost of a card (typically referred to as mana cost) is
usually associated with power - the higher the cost, the stronger the
card. This is because mana cost has an inherent restriction: it takes
time to grow our mana enough to actually cast higher costing cards. We
can see this difference in the cards attached below:

<figure>
<p><img src="/assets/magic-optimizer/2xm-127-goblin-guide.jpeg" alt="image" width="40%"/> 
<img src="/assets/magic-optimizer/ima-149-thundermaw-hellkite.jpeg" alt="image" width="40%" /></p>
<figcaption>Two threats of varying strength and cost</figcaption>
</figure>

First, let’s examine the card on the left, *Goblin Guide*. In the top
right corner, it costs one red mana, so we say it costs one. Moving to
the bottom right corner, the first number corresponds with attack and
the second number corresponds with toughness. Attack indicates how much
damage the creature does while toughness reveals how much damage the
creature can take before dying. In this case, *Goblin Guide* has two
attack and two toughness.  
In a similar manner, we look at *Thundermaw Hellkite*. It costs five
total mana because it has two red pips and an additional generic cost of
three. It has five attack and five toughness.

Clearly, once in play and ignoring abilities, *Thundermaw Hellkite* is
the stronger card because it has better stats. However, it does cost
more, which means we are less likely to actually see it in play.

This is the ambiguity we wish to resolve. How many large cards, like
*Thundermaw Hellkite*, can we get away with in a deck? How many small
cards, like *Goblin Guide*, do we need to fill in the gaps?  
When we begin a game, we look at seven cards as our opening hand. If we
decide they are good enough, we can keep the hand. Otherwise, we take a
mulligan. This mulligan entails shuffling the original hand back into
the deck, and drawing a new hand of seven cards. Once we decide to keep,
we must put cards onto the bottom of our deck equal to the number of
mulligans taken. Here’s a example of a player, called "Avery", resolving
a mulligan:

1.  Avery draws seven cards.

2.  Avery thinks these cards are bad and decides to mulligan.

3.  Upon mulliganing, Avery shuffles that hand into their deck and draws
    seven new cards.

4.  Avery thinks these cards are good and decides to keep.

5.  Upon keeping, Avery has mulliganed once and must therefore pick one
    card from the current seven cards and puts it onto the bottom of
    their deck.

For the purpose of this paper, minimizing mulligans will be one of the
central heuristics. Choosing mulligans is powerful because going down a
card is punishing and hence decreases win rate. Additionally,
mulliganing is a binary, meaning it is easy to train a neural network
on. It would be very, very difficult and not useful to decide whether
each possible opening hand is a keep or mulligan. Firstly, there are
more than 20,000 unique Magic cards, making nearly impossible to
consider finding all the unique hands and classifying them one by one.
Secondly, different decks play different types of cards.

Hence, we will make two simplifications. First, we will consider one
specific deck. This deck will be an aggressively slanted deck trying to
maximize its mana usage in early turns. I picked this deck type because
of its linear game plan, making it easier to see if a hand can be kept
or mulliganed. The second reason to pick this deck bleeds into the
second simplification: there are fundamentally only nine card types in
this deck. We can boil all aggressive cards into the following achetypal
cards:

1.  Lands

2.  Nonland, zero cost mana sources

3.  Nonland, one cost mana sources

4.  One mana threats

5.  Two mana threats

6.  Three mana threats

7.  Four mana threats

8.  Five mana threats

9.  Interaction

The first three items are mana sources. These provide mana, which can
pay costs. Each mana source provides one mana per turn, every turn.  
Lands are mana sources which can be played once per turn.  
Nonland mana sources are similar, but they aren’t constrained to being
played once per turn. Additionally, one cost mana sources cannot be used
to pay costs right away. Lands and zero cost mana sources do not have
this restriction and can be used to pay costs the turn they are
played.

Threats are proactive plays which are actively trying to beat the
opponent. *Goblin Guide* and *Thundermaw Hellkite* above are both
examples of threats.

Interaction classifies cards that remove opposing threats and in general
disrupts what our opponent is doing.

# Data Collection

With these simplifications and heuristics laid out, we can get to work.
As mentioned in the opening, we need to collect data points by hand. To
do this, I wrote a light weight react app which presents a user with a
sample hand. Then, the user can either select "Keep" or "Mulligan."

<figure id="fig:my_label">
<img src="/assets/magic-optimizer/sample_hand_react.png" width="60%"/>
<figcaption>Sample hand generated by the react app</figcaption>
</figure>

Their decision along with the cards in the hand is appended to a .csv
file, to be parsed later for training a neural network.  
It’s also worth noting that the cards are weighted according to what we
think might be a good ratio between archetypes based on our deck
building experience. This means that when we go to use the neural
network, its training data will more closely line up with what we show
it.  
We use this to train a neural network using pandas and the tensorflow
python packages.

# Algorithm Description

We have *n* deck configurations. For the *i*<sup>*th*</sup> deck, we
have
*X*<sub>*i*</sub> ∼ *Beta*(*α*<sub>*i*</sub>,*β*<sub>*i*</sub>).
We initialize *α*<sub>*i*</sub> = 1 and *β*<sub>*i*</sub> = 1. During
each loop, we sample each *X*<sub>*i*</sub> and store the index *j* such
that *p̂*<sub>*j*</sub>=
max<sub>*i* ∈ (1,⋯,*n*)</sub>(*p̂*<sub>*i*</sub>). If there is a tie, we
ignore the new data point. Since our specific use case has a large *n*,
we will only sample 100 random decks. Past the 100<sup>*th*</sup>
deck, it becomes nearly impossible to surpass *p̂*<sub>*j*</sub> so we’d
end up doing a lot of unnecessary sampling.

Next, after sampling each beta distribution, we take a sample hand from
*j*<sup>*th*</sup> deck. Our neural network then decides if that hand
is a keep or mulligan. A keep is a success, and we would have that
*X*<sub>*j*</sub> ∼ *Beta*(*α*<sub>*j*</sub>+1,*β*<sub>*j*</sub>).
A mulligan is a failure, and we would have that
*X*<sub>*j*</sub> ∼ *Beta*(*α*<sub>*j*</sub>,*β*<sub>*j*</sub>+1).
Upon updating *X*<sub>*j*</sub>, we loop again.

We continue in this way until the algorithm has converged to an optimal
deck configuration.

In code, this looks like:

    for t in range(num_iter):
        max_sample = 0
        arm = ""
        # shuffle the keys so the first keys we view are random
        # we do this since the threshold gets so high for MAB that it is basically impossible to surpass
        # past ~100 keys
        random.shuffle(keys)
        for i in range(keys_to_look_at):
            key = keys[i]
            alpha, beta = int(data[key][0]), int(data[key][1])
            s = numpy.random.beta(alpha, beta)
            if max_sample < s:
                arm = key
                max_sample = s

        sample_hand = get_sample_hand(arm)
        prediction = model.predict(numpy.array([sample_hand, ]), verbose=0)
        if prediction >= 0.5:
            data[arm][0] += 1
        else:
            data[arm][1] += 1

        # write every so often to we can continually store data so our run won't be lost if it stops
        if t % itr_to_store_data == 0:
            write_small_subset_data(data)

# Deck generation

The number of possible decks is 10<sup>100</sup>, which is way too large
to run this MAB on given the computing power I have available. For that
reason, we limit the number of minimum and maximum cards for each type
in the deck. To start, we will have

1.  31 to 36 lands

2.  2 zero mana accelerators

3.  10 to 15 one mana accelerators

4.  8 to 16 one mana threats

5.  10 to 17 two mana threats

6.  6 to 15 three mana threats

7.  5 to 15 four mana threats

8.  0 five mana threats

9.  8 pieces of interaction

These values, including the fixed ones, are generated based on personal
experience. In the future, we can refine these numbers based on what the
MAB algorithm converges on.  
With these numbers, we use a memoized, recursive solution to generate
all possible decks satisfying these constraints.  
In code, this solution looks like

    base_deck, min_deck_size = get_base_deck()
    set_of_decks = set()

    def generate_decks(curr_deck_so_far, types, which_type, max_in_deck):
        cards_in_deck = sum(curr_deck_so_far)
        if tuple(curr_deck_so_far) in set_of_decks or cards_in_deck > max_in_deck:
            return

        if cards_in_deck == max_in_deck:
            set_of_decks.add(tuple(curr_deck_so_far))

        if cards_in_deck < max_in_deck and which_type < len(types):
            for i in range(card_types[types[which_type]]['range']):
                # this makes a new instance of the array,
                # meaning the future recursive calls won't touch this same array
                tmp = curr_deck_so_far[:]
                curr_deck_so_far[which_type] = card_types[types[which_type]]
                                                ['min_number'] + i
                generate_decks(curr_deck_so_far,
                               types,
                               which_type + 1,
                               max_in_deck)
                curr_deck_so_far = tmp}

# Results

This algorithm converged on the deck configuration:

1.  32 lands

2.  2 zero mana accelerators

3.  14 one mana accelerators

4.  14 one mana threats

5.  17 two mana threats

6.  7 three mana threats

7.  6 four mana threats

8.  0 five mana threats

9.  8 interactions

This result made sense. It seems like it wouldn’t mulligan that much
since it is really heavy on smaller threats and accelerators. In
essence, it is optimizing for the early turns of the game, which is what
someone making a mulligan decision can see. It is unclear if this is
actually the best deck, even if it minimizes the mulligan chance.  
The methodology for creating this deck also has a number of
improvements.  
First, the neural network used to predict this likely had too many
nodes. I used three layers - one with 256 nodes, one with 128 nodes, and
an output layer for classification. Since we only had 600-700 data
points, it is likely this neural network was way to big. It could have
built each data point into the neural network.  
Additionally, the cutoff for keep and mulligan was set arbitrarily at
0.5. While there is nothing inherently wrong with this cutoff, we may do
better when setting it to a different number. For reference, here is the
ROC for this neural network:

<figure id="fig:my_label">
<div class="center">
<img src="/assets/magic-optimizer/keras_roc_plot.png" />
</div>
<figcaption>ROC Graph for Keras Neural Network</figcaption>
</figure>

Based on this graph, it seems like we could do better by increasing the
cutoff.

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/