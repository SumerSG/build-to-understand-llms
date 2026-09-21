// lib/data.js — the lab's datasets: a procedural toy corpus, public-domain prose, instruction and
// preference pairs, verifiable arithmetic tasks, the chat template, and batching helpers.
//
// Everything here is generated deterministically from lib/util.js's seeded rng, so two runs of the lab
// see byte-identical data. The corpus is deliberately small (about 40 KB): it has to tokenize and train
// inside a browser tab in seconds, and a ~100k-parameter model can only learn structure that is repeated
// often, which is exactly what the toy grammar provides.

import { rng, randInt, choice } from './util.js';

// Nouns carry both numbers so the templates below can keep subject-verb agreement.
const SUBJECTS = [
  { one: 'cat', many: 'cats' }, { one: 'dog', many: 'dogs' }, { one: 'bird', many: 'birds' },
  { one: 'fox', many: 'foxes' }, { one: 'mouse', many: 'mice' }, { one: 'child', many: 'children' },
  { one: 'teacher', many: 'teachers' }, { one: 'farmer', many: 'farmers' }, { one: 'sailor', many: 'sailors' },
  { one: 'robot', many: 'robots' }, { one: 'girl', many: 'girls' }, { one: 'boy', many: 'boys' },
  { one: 'doctor', many: 'doctors' }, { one: 'baker', many: 'bakers' }, { one: 'painter', many: 'painters' },
  { one: 'student', many: 'students' },
];

// Verbs that take no object: "the cat sleeps".
const VERBS = [
  { one: 'sleeps', many: 'sleep' }, { one: 'runs', many: 'run' }, { one: 'sings', many: 'sing' },
  { one: 'waits', many: 'wait' }, { one: 'walks', many: 'walk' }, { one: 'laughs', many: 'laugh' },
  { one: 'works', many: 'work' }, { one: 'rests', many: 'rest' }, { one: 'listens', many: 'listen' },
  { one: 'dances', many: 'dance' },
];

// Verbs that take an object: "the cat chases the ball".
const TRANSITIVE = [
  { one: 'chases', many: 'chase' }, { one: 'finds', many: 'find' }, { one: 'carries', many: 'carry' },
  { one: 'paints', many: 'paint' }, { one: 'reads', many: 'read' }, { one: 'wants', many: 'want' },
  { one: 'hides', many: 'hide' }, { one: 'counts', many: 'count' }, { one: 'sells', many: 'sell' },
  { one: 'watches', many: 'watch' },
];

const OBJECTS = [
  { one: 'ball', many: 'balls' }, { one: 'book', many: 'books' }, { one: 'apple', many: 'apples' },
  { one: 'letter', many: 'letters' }, { one: 'basket', many: 'baskets' }, { one: 'hat', many: 'hats' },
  { one: 'map', many: 'maps' }, { one: 'key', many: 'keys' }, { one: 'lamp', many: 'lamps' },
  { one: 'boat', many: 'boats' }, { one: 'song', many: 'songs' }, { one: 'garden', many: 'gardens' },
];

const ADJECTIVES = [
  'small', 'big', 'red', 'blue', 'green', 'quiet', 'quick', 'happy', 'sleepy', 'old', 'new', 'brave',
  'warm', 'clever', 'hungry', 'kind',
];

const ADVERBS = [
  'quietly', 'quickly', 'slowly', 'gladly', 'softly', 'often', 'again', 'today', 'twice', 'loudly',
];

const PLACES = [
  'near the river', 'under the tree', 'in the garden', 'on the hill', 'by the lake', 'behind the barn',
  'beside the road', 'in the kitchen', 'over the bridge', 'at the market',
];

const NUMBER_WORDS = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];

/** Uppercase the first letter, so every sentence starts the way English sentences do. */
function capitalize(sentence) {
  return sentence.charAt(0).toUpperCase() + sentence.slice(1);
}

/** One sentence from the toy grammar. The template mix is what gives the corpus its statistics. */
function toySentence(next) {
  const subject = choice(next, SUBJECTS);
  const other = choice(next, SUBJECTS);
  const verb = choice(next, VERBS);
  const verb2 = choice(next, VERBS);
  const transitive = choice(next, TRANSITIVE);
  const object = choice(next, OBJECTS);
  const adjective = choice(next, ADJECTIVES);
  const adverb = choice(next, ADVERBS);
  const place = choice(next, PLACES);
  const word = choice(next, NUMBER_WORDS);
  const digits = randInt(next, 9) + 2;
  switch (randInt(next, 10)) {
    case 0: return `the ${adjective} ${subject.one} ${verb.one} ${adverb}.`;
    case 1: return `the ${adjective} ${subject.many} ${verb.many} ${place}.`;
    case 2: return `a ${subject.one} ${transitive.one} the ${adjective} ${object.one} ${place}.`;
    case 3: return `${word} ${subject.many} ${transitive.many} ${digits} ${object.many}.`;
    case 4: return `the ${subject.one} ${verb.one} ${adverb}, and the ${other.one} ${verb2.one} too.`;
    case 5: return `when the ${subject.one} ${verb.one}, the ${other.many} ${verb2.many} ${place}.`;
    case 6: return `where did the ${adjective} ${subject.one} go?`;
    case 7: return `the ${subject.one} is ${adjective} ${place}.`;
    case 8: return `${digits} ${object.many} are ${adjective}, but the ${subject.one} ${transitive.one} ${word}.`;
    default: return `the ${subject.many} ${transitive.many} ${digits} ${adjective} ${object.many} ${place}.`;
  }
}

/** A procedural English-like corpus: one sentence per line, deterministic for a given seed. */
export function toyCorpus(nSentences = 4000, seed = 1) {
  const next = rng(seed);
  const lines = new Array(nSentences);
  for (let i = 0; i < nSentences; i++) lines[i] = capitalize(toySentence(next));
  return lines.join('\n');
}

// Public-domain English, transcribed with ASCII punctuation so the character vocabulary stays small.
// Sources, all long out of copyright:
//   - William Shakespeare, Sonnets 18, 29, 55, 73, 116 and 130 (first published 1609).
//   - Lewis Carroll, "Alice's Adventures in Wonderland" (1865), opening of Chapter 1.
//   - Jane Austen, "Pride and Prejudice" (1813), opening of Chapter 1.
//   - Charles Dickens, "A Tale of Two Cities" (1859), opening paragraph.
//   - Herman Melville, "Moby-Dick" (1851), opening paragraph.
//   - L. Frank Baum, "The Wonderful Wizard of Oz" (1900), opening of Chapter 1.
//   - William Blake, "The Tyger", from Songs of Experience (1794).
export const PROSE = `Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date;
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.

When, in disgrace with fortune and men's eyes,
I all alone beweep my outcast state,
And trouble deaf heaven with my bootless cries,
And look upon myself and curse my fate,
Wishing me like to one more rich in hope,
Featured like him, like him with friends possess'd,
Desiring this man's art and that man's scope,
With what I most enjoy contented least;
Yet in these thoughts myself almost despising,
Haply I think on thee, and then my state,
Like to the lark at break of day arising
From sullen earth, sings hymns at heaven's gate;
For thy sweet love remember'd such wealth brings
That then I scorn to change my state with kings.

Not marble, nor the gilded monuments
Of princes, shall outlive this powerful rhyme;
But you shall shine more bright in these contents
Than unswept stone besmear'd with sluttish time.
When wasteful war shall statues overturn,
And broils root out the work of masonry,
Nor Mars his sword nor war's quick fire shall burn
The living record of your memory.
'Gainst death and all-oblivious enmity
Shall you pace forth; your praise shall still find room
Even in the eyes of all posterity
That wear this world out to the ending doom.
So, till the judgment that yourself arise,
You live in this, and dwell in lovers' eyes.

That time of year thou mayst in me behold
When yellow leaves, or none, or few, do hang
Upon those boughs which shake against the cold,
Bare ruin'd choirs, where late the sweet birds sang.
In me thou see'st the twilight of such day
As after sunset fadeth in the west,
Which by and by black night doth take away,
Death's second self, that seals up all in rest.
In me thou see'st the glowing of such fire
That on the ashes of his youth doth lie,
As the death-bed whereon it must expire,
Consumed with that which it was nourish'd by.
This thou perceivest, which makes thy love more strong,
To love that well which thou must leave ere long.

Let me not to the marriage of true minds
Admit impediments. Love is not love
Which alters when it alteration finds,
Or bends with the remover to remove:
O no! it is an ever-fixed mark
That looks on tempests and is never shaken;
It is the star to every wandering bark,
Whose worth's unknown, although his height be taken.
Love's not Time's fool, though rosy lips and cheeks
Within his bending sickle's compass come;
Love alters not with his brief hours and weeks,
But bears it out even to the edge of doom.
If this be error and upon me proved,
I never writ, nor no man ever loved.

My mistress' eyes are nothing like the sun;
Coral is far more red than her lips' red;
If snow be white, why then her breasts are dun;
If hairs be wires, black wires grow on her head.
I have seen roses damask'd, red and white,
But no such roses see I in her cheeks;
And in some perfumes is there more delight
Than in the breath that from my mistress reeks.
I love to hear her speak, yet well I know
That music hath a far more pleasing sound;
I grant I never saw a goddess go;
My mistress, when she walks, treads on the ground:
And yet, by heaven, I think my love as rare
As any she belied with false compare.

Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do:
once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations
in it, "and what is the use of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and
stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking
the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear
the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" But when the Rabbit actually took a watch
out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it
flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch
to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just
in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out
again.

The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly
that Alice had not a moment to think about stopping herself before she found herself falling down a very
deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to
look about her and to wonder what was going to happen next. First, she tried to look down and make out
what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and
noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures
hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled "ORANGE
MARMALADE", but to her great disappointment it was empty: she did not like to drop the jar for fear of
killing somebody underneath, so managed to put it into one of the cupboards as she fell past it.

"Well!" thought Alice to herself, "after such a fall as this, I shall think nothing of tumbling down
stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off
the top of the house!"

Down, down, down. Would the fall never come to an end? "I wonder how many miles I've fallen by this time?"
she said aloud. "I must be getting somewhere near the centre of the earth. Let me see: that would be four
thousand miles down, I think."

It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want
of a wife.

However little known the feelings or views of such a man may be on his first entering a neighbourhood,
this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful
property of some one or other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park is let at last?"

Mr. Bennet replied that he had not.

"But it is," returned she; "for Mrs. Long has just been here, and she told me all about it."

Mr. Bennet made no answer.

"Do you not want to know who has taken it?" cried his wife impatiently.

"You want to tell me, and I have no objection to hearing it."

This was invitation enough.

"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune
from the north of England; that he came down on Monday in a chaise and four to see the place, and was so
much delighted with it that he agreed with Mr. Morris immediately; that he is to take possession before
Michaelmas, and some of his servants are to be in the house by the end of next week."

It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of
foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it
was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything
before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the
other way.

Call me Ishmael. Some years ago -- never mind how long precisely -- having little or no money in my purse,
and nothing particular to interest me on shore, I thought I would sail about a little and see the watery
part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I
find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I
find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I
meet; then, I account it high time to get to sea as soon as I can.
Presently she began again. "I wonder if I shall fall right through the earth! How funny it'll seem to come
out among the people that walk with their heads downward! But I shall have to ask them what the name of
the country is, you know. Please, Ma'am, is this New Zealand or Australia?" And she tried to curtsey as
she spoke -- fancy curtseying as you're falling through the air! Do you think you could manage it? "And
what an ignorant little girl she'll think me for asking! No, it'll never do to ask: perhaps I shall see it
written up somewhere."

Down, down, down. There was nothing else to do, so Alice soon began talking again. "Dinah'll miss me very
much to-night, I should think!" Dinah was the cat. "I hope they'll remember her saucer of milk at
tea-time. Dinah my dear! I wish you were down here with me! There are no mice in the air, I'm afraid, but
you might catch a bat, and that's very like a mouse, you know. But do cats eat bats, I wonder?" And here
Alice began to get rather sleepy, and went on saying to herself, in a dreamy sort of way, "Do cats eat
bats? Do cats eat bats?" and sometimes, "Do bats eat cats?" for, you see, as she couldn't answer either
question, it didn't much matter which way she put it.

Dorothy lived in the midst of the great Kansas prairies, with Uncle Henry, who was a farmer, and Aunt Em,
who was the farmer's wife. Their house was small, for the lumber to build it had to be carried by wagon
many miles. There were four walls, a floor and a roof, which made one room; and this room contained a
rusty looking cookstove, a cupboard for the dishes, a table, three or four chairs, and the beds. Uncle
Henry and Aunt Em had a big bed in one corner, and Dorothy a little bed in another corner. There was no
garret at all, and no cellar -- except a small hole dug in the ground, called a cyclone cellar, where the
family could go in case one of those great whirlwinds arose, mighty enough to crush any building in its
path.

Tyger Tyger, burning bright,
In the forests of the night;
What immortal hand or eye,
Could frame thy fearful symmetry?

In what distant deeps or skies,
Burnt the fire of thine eyes?
On what wings dare he aspire?
What the hand, dare seize the fire?

And what shoulder, and what art,
Could twist the sinews of thy heart?
And when thy heart began to beat,
What dread hand? and what dread feet?

What the hammer? what the chain,
In what furnace was thy brain?
What the anvil? what dread grasp,
Dare its deadly terrors clasp!

When the stars threw down their spears
And water'd heaven with their tears:
Did he smile his work to see?
Did he who made the Lamb make thee?

Tyger Tyger burning bright,
In the forests of the night:
What immortal hand or eye,
Dare frame thy fearful symmetry?`;

// How many toy sentences go into CORPUS. LIB_API.md asks for a roughly 40 KB CORPUS; readable sentences
// average about 35 characters, so 800 of them plus PROSE lands in that budget (see the report note).
const CORPUS_SENTENCES = 800;

/** The pre-training corpus: procedural toy sentences first, then real public-domain English. */
export const CORPUS = `${toyCorpus(CORPUS_SENTENCES, 1)}\n\n${PROSE}`;

/** Short instruction/response pairs in the toy vocabulary, for supervised fine-tuning (module 10). */
export const INSTRUCTIONS = [
  { prompt: 'Say hello.', response: 'Hello!' },
  { prompt: 'Say goodbye.', response: 'Goodbye!' },
  { prompt: 'Greet the teacher.', response: 'Hello, teacher!' },
  { prompt: 'Name a color.', response: 'Red.' },
  { prompt: 'Name another color.', response: 'Blue.' },
  { prompt: 'Name a warm color.', response: 'Red is a warm color.' },
  { prompt: 'Name an animal.', response: 'A cat.' },
  { prompt: 'Name a small animal.', response: 'A mouse is a small animal.' },
  { prompt: 'Name a big animal.', response: 'A horse is a big animal.' },
  { prompt: 'Name a bird.', response: 'A bird I like is the owl.' },
  { prompt: 'Count to three.', response: 'One, two, three.' },
  { prompt: 'Count to five.', response: 'One, two, three, four, five.' },
  { prompt: 'Count backward from three.', response: 'Three, two, one.' },
  { prompt: 'Repeat: the cat sat.', response: 'The cat sat.' },
  { prompt: 'Repeat: the dog runs.', response: 'The dog runs.' },
  { prompt: 'Repeat: birds sing.', response: 'Birds sing.' },
  { prompt: 'Repeat: the boat is red.', response: 'The boat is red.' },
  { prompt: 'What is two plus two?', response: 'Four.' },
  { prompt: 'What is three plus four?', response: 'Seven.' },
  { prompt: 'What is ten minus one?', response: 'Nine.' },
  { prompt: 'What is two times three?', response: 'Six.' },
  { prompt: 'What color is the sky?', response: 'The sky is blue.' },
  { prompt: 'What color is grass?', response: 'Grass is green.' },
  { prompt: 'What color is snow?', response: 'Snow is white.' },
  { prompt: 'What sound does a cat make?', response: 'A cat says meow.' },
  { prompt: 'What sound does a dog make?', response: 'A dog says woof.' },
  { prompt: 'Where does a fish live?', response: 'A fish lives in the river.' },
  { prompt: 'Where does a bird sleep?', response: 'A bird sleeps in the tree.' },
  { prompt: 'Where does the farmer work?', response: 'The farmer works in the garden.' },
  { prompt: 'Where does the baker work?', response: 'The baker works in the kitchen.' },
  { prompt: 'Who reads books?', response: 'A student reads books.' },
  { prompt: 'Who paints pictures?', response: 'A painter paints pictures.' },
  { prompt: 'Who sails the boat?', response: 'A sailor sails the boat.' },
  { prompt: 'Who helps sick people?', response: 'A doctor helps sick people.' },
  { prompt: 'Write a short sentence about a cat.', response: 'The small cat sleeps quietly.' },
  { prompt: 'Write a short sentence about a dog.', response: 'The big dog runs in the garden.' },
  { prompt: 'Write a short sentence about a bird.', response: 'The quick bird sings near the river.' },
  { prompt: 'Write a short sentence about the market.', response: 'The farmers sell apples at the market.' },
  { prompt: 'Write a question about a boat.', response: 'Where did the old boat go?' },
  { prompt: 'Write a question about a book.', response: 'Who reads the new book?' },
  { prompt: 'Make the sentence longer: the cat sat.', response: 'The small cat sat under the tree.' },
  { prompt: 'Make the sentence shorter: the happy dog runs quickly in the garden.', response: 'The dog runs.' },
  { prompt: 'Turn this into a question: the bird sings.', response: 'Does the bird sing?' },
  { prompt: 'Turn this into a question: the boat is red.', response: 'Is the boat red?' },
  { prompt: 'Make this plural: cat.', response: 'Cats.' },
  { prompt: 'Make this plural: child.', response: 'Children.' },
  { prompt: 'Make this plural: mouse.', response: 'Mice.' },
  { prompt: 'Make this singular: dogs.', response: 'Dog.' },
  { prompt: 'Give the opposite of big.', response: 'Small.' },
  { prompt: 'Give the opposite of old.', response: 'New.' },
  { prompt: 'Give the opposite of quick.', response: 'Slow.' },
  { prompt: 'Give the opposite of happy.', response: 'Sad.' },
  { prompt: 'List two colors.', response: 'Red and blue.' },
  { prompt: 'List three animals.', response: 'A cat, a dog, and a bird.' },
  { prompt: 'List two places.', response: 'The garden and the market.' },
  { prompt: 'Finish the sentence: the cat sat on the', response: 'The cat sat on the hat.' },
  { prompt: 'Finish the sentence: the birds sing near the', response: 'The birds sing near the river.' },
  { prompt: 'Spell the word cat.', response: 'C, a, t.' },
  { prompt: 'Spell the word dog.', response: 'D, o, g.' },
  { prompt: 'How many legs does a cat have?', response: 'A cat has four legs.' },
  { prompt: 'How many days are in a week?', response: 'Seven days.' },
  { prompt: 'Tell me something a robot can do.', response: 'A robot can carry baskets.' },
  { prompt: 'Tell me something a teacher can do.', response: 'A teacher can read a book to the children.' },
  { prompt: 'Say thank you.', response: 'Thank you!' },
  { prompt: 'Say please.', response: 'Please!' },
  { prompt: 'Ask the baker for bread.', response: 'Baker, may I have some bread, please?' },
];

/** Preference triples for reward modelling and DPO (module 11): concise and on topic beats the rest. */
export const PREFERENCES = [
  { prompt: 'Say hello.', chosen: 'Hello!', rejected: 'The boat is red and the market is far away.' },
  { prompt: 'Name a color.', chosen: 'Red.', rejected: 'A cat is an animal that sleeps in the garden.' },
  { prompt: 'Name an animal.', chosen: 'A cat.', rejected: 'Blue.' },
  { prompt: 'Count to three.', chosen: 'One, two, three.', rejected: 'One, two,' },
  { prompt: 'Count to five.', chosen: 'One, two, three, four, five.', rejected: 'One, two, three' },
  { prompt: 'Repeat: the cat sat.', chosen: 'The cat sat.', rejected: 'The dog ran.' },
  { prompt: 'Repeat: birds sing.', chosen: 'Birds sing.', rejected: 'Birds' },
  { prompt: 'What color is the sky?', chosen: 'The sky is blue.', rejected: 'The sky is a thing that is above the hill and the river and the barn and the road.' },
  { prompt: 'What color is grass?', chosen: 'Grass is green.', rejected: 'Grass is red.' },
  { prompt: 'What is two plus two?', chosen: 'Four.', rejected: 'Five.' },
  { prompt: 'What is three plus four?', chosen: 'Seven.', rejected: 'The answer is a number near seven, or maybe eight, it is hard to say.' },
  { prompt: 'What is ten minus one?', chosen: 'Nine.', rejected: 'Ten.' },
  { prompt: 'Where does a fish live?', chosen: 'A fish lives in the river.', rejected: 'A fish lives in the kitchen.' },
  { prompt: 'Where does a bird sleep?', chosen: 'A bird sleeps in the tree.', rejected: 'A bird' },
  { prompt: 'Where does the baker work?', chosen: 'The baker works in the kitchen.', rejected: 'The sailor works on the boat, and the farmer works in the garden, and the painter paints.' },
  { prompt: 'Who reads books?', chosen: 'A student reads books.', rejected: 'A book reads a student.' },
  { prompt: 'Who sails the boat?', chosen: 'A sailor sails the boat.', rejected: 'The boat sails the' },
  { prompt: 'Who paints pictures?', chosen: 'A painter paints pictures.', rejected: 'Red and blue and green.' },
  { prompt: 'Write a short sentence about a cat.', chosen: 'The small cat sleeps quietly.', rejected: 'The cat the cat the cat the cat the cat the cat.' },
  { prompt: 'Write a short sentence about a dog.', chosen: 'The big dog runs in the garden.', rejected: 'Dog.' },
  { prompt: 'Write a question about a boat.', chosen: 'Where did the old boat go?', rejected: 'The old boat went away.' },
  { prompt: 'Turn this into a question: the bird sings.', chosen: 'Does the bird sing?', rejected: 'The bird sings.' },
  { prompt: 'Make this plural: cat.', chosen: 'Cats.', rejected: 'Cat.' },
  { prompt: 'Make this plural: child.', chosen: 'Children.', rejected: 'Childs.' },
  { prompt: 'Make this plural: mouse.', chosen: 'Mice.', rejected: 'Mouses.' },
  { prompt: 'Make this singular: dogs.', chosen: 'Dog.', rejected: 'Dogs are animals that run in the garden near the river.' },
  { prompt: 'Give the opposite of big.', chosen: 'Small.', rejected: 'Big.' },
  { prompt: 'Give the opposite of old.', chosen: 'New.', rejected: 'The opposite is a word that means not old, and there are many such words.' },
  { prompt: 'Give the opposite of quick.', chosen: 'Slow.', rejected: 'Quickly.' },
  { prompt: 'List two colors.', chosen: 'Red and blue.', rejected: 'Red.' },
  { prompt: 'List three animals.', chosen: 'A cat, a dog, and a bird.', rejected: 'A cat and a dog.' },
  { prompt: 'Finish the sentence: the cat sat on the', chosen: 'The cat sat on the hat.', rejected: 'The cat sat on the' },
  { prompt: 'Finish the sentence: the birds sing near the', chosen: 'The birds sing near the river.', rejected: 'Birds are small and they have wings and they sing.' },
  { prompt: 'Spell the word cat.', chosen: 'C, a, t.', rejected: 'Cat.' },
  { prompt: 'How many legs does a cat have?', chosen: 'A cat has four legs.', rejected: 'A cat has legs.' },
  { prompt: 'How many days are in a week?', chosen: 'Seven days.', rejected: 'Many days.' },
  { prompt: 'Tell me something a robot can do.', chosen: 'A robot can carry baskets.', rejected: 'A robot is a robot is a robot is a robot.' },
  { prompt: 'Say thank you.', chosen: 'Thank you!', rejected: 'Please!' },
  { prompt: 'Say goodbye.', chosen: 'Goodbye!', rejected: 'Hello!' },
  { prompt: 'Greet the teacher.', chosen: 'Hello, teacher!', rejected: 'The teacher walks to the market with five baskets and a map.' },
  { prompt: 'Ask the baker for bread.', chosen: 'Baker, may I have some bread, please?', rejected: 'Bread.' },
  { prompt: 'Name a small animal.', chosen: 'A mouse is a small animal.', rejected: 'A mouse is a big animal.' },
];

/** One-step arithmetic with checkable answers, for RL with verifiable rewards (module 12). */
function makeMathTasks(count = 120) {
  const next = rng(3);
  const tasks = [];
  while (tasks.length < count) {
    const op = choice(next, ['+', '-', '*']);
    let a = randInt(next, 12) + 1;
    let b = randInt(next, 12) + 1;
    if (op === '*') { a = randInt(next, 9) + 1; b = randInt(next, 9) + 1; }
    if (op === '-' && b > a) { const t = a; a = b; b = t; } // keep answers non-negative
    const answer = op === '+' ? a + b : op === '-' ? a - b : a * b;
    tasks.push({ question: `What is ${a} ${op} ${b}?`, answer: String(answer) });
  }
  return tasks;
}

/** 120 arithmetic tasks, generated once at import so every module sees the same list. */
export const MATH_TASKS = makeMathTasks();

/** The chat markers the lab's templates and checkpoints use. */
export const CHAT = { system: '<|system|>', user: '<|user|>', assistant: '<|assistant|>', end: '<|end|>' };

/** Render messages as one prompt string; a trailing user turn opens the assistant's turn for the model. */
export function formatChat(messages) {
  let out = '';
  for (const message of messages) {
    const marker = CHAT[message.role] ?? CHAT.user;
    out += marker + message.content + CHAT.end;
  }
  const last = messages[messages.length - 1];
  if (last && last.role === 'user') out += CHAT.assistant;
  return out;
}

/**
 * A batch of training windows: x is blockSize ids starting at a random position, y is the same window
 * shifted one to the left, so position t of y is the token the model must predict from x[0..t].
 */
export function getBatch(ids, { blockSize, batchSize, next }) {
  const lastStart = ids.length - blockSize - 1;
  if (lastStart < 0) throw new Error(`getBatch: need at least ${blockSize + 1} ids, got ${ids.length}`);
  const x = [];
  const y = [];
  for (let b = 0; b < batchSize; b++) {
    const start = randInt(next, lastStart + 1);
    const xi = new Array(blockSize);
    const yi = new Array(blockSize);
    for (let t = 0; t < blockSize; t++) {
      xi[t] = ids[start + t];
      yi[t] = ids[start + t + 1];
    }
    x.push(xi);
    y.push(yi);
  }
  return { x, y };
}

/** Split a token stream into train/validation by position: the tail is held out, never shuffled. */
export function trainValSplit(ids, frac = 0.9) {
  const cut = Math.floor(ids.length * frac);
  return { train: ids.slice(0, cut), val: ids.slice(cut) };
}

/**
 * Hold out every `holdOut`-th chunk of `chunk` tokens as validation. Unlike the contiguous
 * trainValSplit, this keeps the validation set representative when the corpus is a concatenation
 * of different sources (the toy sentences and the public-domain prose).
 */
export function interleavedSplit(ids, { chunk = 256, holdOut = 10 } = {}) {
  const train = [];
  const val = [];
  for (let start = 0, k = 0; start < ids.length; start += chunk, k++) {
    const piece = ids.slice(start, start + chunk);
    if (k % holdOut === holdOut - 1) val.push(...piece); else train.push(...piece);
  }
  return { train, val };
}
