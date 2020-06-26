# conllulex2ucca.py (2020 version): rule-based converter overview

_Nathan Schneider_

This gives a step-by-step overview of the algorithm for constructing
an UCCA semantic graph using STREUSLE/UD annotations.

It is not a full specification and omits many details of the criteria and operations,
but should be helpful for understanding the code.

## Step 0.1: Transform the UD dependency parse

- Deal with some issues with coordination and parataxis (Daniel, do you remember the rationale?) via DepEdit rules.
- Split the final preposition off from `V.IAV` MWEs like *take care of*, as it is usually not treated as part of the verbal unit in UCCA. If the remaining part is still an MWE, it is labeled `V.LVC.full` or `V.VPC.full` depending on its syntax.

## Step 0.2: Initialize lexical units under the UCCA root

- Each strong lexical expression (single-word or MWE) in STREUSLE is treated as an UCCA unit with the following exceptions:
  * An MWE with lexcat `V.LVC.cause` is broken into two units: `D` for the light verb modifies the main predicate in a `+` unit.
  * An MWE with lexcat `V.LVC.full` is broken into two units: `F` for the light verb modifies the main predciate in a `+` unit.
  * An MWE annotation is discarded if it would lead to a cycle in the dependencies such that the highest token of the MWE is dependent on a token outside of the MWE, which is dependent on another token of the MWE.
- MWE-internal dependency edges are discarded so they will not be processed later.
- Punctuation is marked `U`.
- Mappings between units and dependency nodes are maintained.

## Step 0.3: Identify which lexical units are main relations (scene-evoking)

In a top-down traversal of the dependency parse, visit each word's lexical unit and decide whether it evokes a state (`S`), process (`P`), undetermined between state or process (`+`), or does not evoke a scene (`-`):

- If already labeled `D`, `F`, or `+`, do nothing.
- If an adjective not from a small list of quantity adjectives, label `S`.
- If existential *there*, label `S`. If a *be* verb in an existential construction, label `-` and swap the positions of *be* and *there* in the dependency parse so *there* is the head.
- If an adverb not attached as discourse and it has a copula, label `S`.
- *thanks* and *thank you* are `P`.
- In most cases, predicative prepositions are `S`.
- A copula introducing a predicate nominal (non-PP) is labeled `S` and promoted to the head of the dependency parse, unless the nominal is scene-evoking. (The top-down traversal order ensures the nominal is reached first.)
- If a common noun, mark as
  * `S` if supersense-tagged as ATTRIBUTE, FEELING, or STATE
  * `P` if ACT, PHENOMENON, PROCESS, or EVENT (with the exception of nouns denoting a part of the day)
  * a relational noun _(TODO: DETAILS)_ if PERSON or GROUP and matching kinship/occupation lists or suffixes
  * `-` otherwise
- If a verb or copula not handled above, label `+`
- Else label `-`

<!-- adjectives, existential "there", and most predicative complements are marked as "S";
        # relational nouns are "A|S" or "A|P" -->

_In the notation, "UNA" means "lexical" (it originally meant "unanalyzable")._

> There's plenty of parking, and I've never had an issue with audience members who won't stop talking or answering their cellphones.

> [DUMMYROOT [S [UNA There] ] [- [UNA 's] ] [- [UNA plenty] ] [- [UNA of] ] [S [UNA parking] ] [U ,] [- [UNA and] ] [- [UNA I] ] [- [UNA 've] ] [- [UNA never] ] [+ [F had] ... [UNA issue] ] [- [UNA an] ] ... [- [UNA with] ] [- [UNA audience] ] [- [UNA|A|P [UNA members] ] ] [- [UNA who] ] [- [UNA wo] ] [- [UNA n't] ] [+ [UNA stop] ] [+ [UNA talking] ] [- [UNA or] ] [+ [UNA answering] ] [- [UNA their] ] [- [UNA cellphones] ] [U .] ]

## Step 1: Attach functional and discourse modifier words

Determiners, auxiliaries, copulas are generally `F`; vocatives and interjections, `G`.
Exceptions include modal auxiliaries (`D`), demonstrative determiners modifying a non-scene unit (`E`), quantifier determiners modifying a non-scene unit (`Q`).

_Omitting the root:_

>  [S [UNA There] [**F** [UNA 's] ] ] [- [UNA plenty] ] [- [UNA of] ] [S [UNA parking] ] [U ,] [- [UNA and] ] [- [UNA I] ]
>  [+ [**F** [UNA 've] ] ... [**F** had] [**F** [UNA an] ] [UNA issue] ] [- [UNA never] ] ... [- [UNA with] ]
>  [- [UNA audience] ] [- [UNA|A|P [UNA members] ] ] [- [UNA who] ] [+ [F [UNA wo] ] ... [UNA stop] ] [- [UNA n't] ] ... [+ [UNA talking] ]
> [- [UNA or] ] [+ [UNA answering] ] [- [UNA their] ] [- [UNA cellphones] ] [U .]

## Step 2: Attach other modifiers: adverbial, adjectival, numeric, compound, possessive, predicative-PP, adnominal-PP; as well as possessive clitic and preposition (as `R`, unless possessive clitic marks canonical possession in which case it is `S`)

> [S [UNA There] [F [UNA 's] ] ] [- [UNA plenty] [**E** [S [**R** [UNA of] ] [UNA parking] ] ] ] [U ,] [- [UNA and] ] [- [UNA I] ] [+ [F [UNA 've] ] [**T** [UNA never] ] [F had] [F [UNA an] ] [UNA issue] [A [**R** [UNA with] ] [**E** [UNA audience] ] [UNA|A|P [UNA members] ] ... [**E** [+ **[A\* members]** [**F** [UNA wo] ] [**D** [UNA n't] ] [UNA stop] ] ] ] ] [- [UNA who] ] ... [+ [UNA talking] ] [- [UNA or] ] [+ [UNA answering] ] [- [**E** [**A|S** [UNA their] ] **[A* cellphones]** ] [UNA cellphones] ] [U .]

## Step 3: Process verbal argument structure relations: subjects, objects, obliques, clausal complements; flag secondary (non-auxiliary) verb constructions

> [S [UNA There] [F [UNA 's] ] [**A** [UNA plenty] [E [S [R [UNA of] ] [UNA parking] ] ] ] ] [U ,] [- [UNA and] ] [+ [**A** [UNA I] ] [F [UNA 've] ] [T [UNA never] ] [F had] [F [UNA an] ] [UNA issue] [A [R [UNA with] ] [E [UNA audience] ] [UNA|A|P [UNA members] ] [E [+ [A\* members] [R [UNA who] ] [F [UNA wo] ] [D [UNA n't] ] [UNA stop] [**^** [+ [UNA talking] ] ] ] ] ] ] [- [UNA or] ] [+ [UNA answering] [**A** [E [A|S [UNA their] ] [A\* cellphones] ] [UNA cellphones] ] ] [U .]

## Step 4: Coordination

Traversing the graph top-down: for each coordinate construction with conjuncts' units' categories X and Y, create a ternary-branching structure `[X(COORD) X L Y]` if X is scene-evoking (`+`, `P`, or `S`) and `[X(COORD) X N Y]` otherwise.

> There's plenty...and I've never had an issue with...

> [**S(COORD)** [S [UNA There] [F [UNA 's] ] [A [UNA plenty] [E [S [R [UNA of] ] [UNA parking] ] ] ] ] ... [**L** [UNA and] ] [+ [A [UNA I] ] [F [UNA 've] ] [T [UNA never] ] [F had] [F [UNA an] ] [UNA issue] [A [R [UNA with] ] [E [UNA audience] ] [UNA|A|P [UNA members] ] [E [+ [A\* members] [R [UNA who] ] [F [UNA wo] ] [D [UNA n't] ] [UNA stop] [^ [+ [UNA talking] ] ] ] ] ] ] ] [U ,] ... [- [UNA or] ] [+ [UNA answering] [A [E [A|S [UNA their] ] [A\* cellphones] ] [UNA cellphones] ] ] [U .]

> ...won't stop talking or answering...

> [S(COORD) [S [UNA There] [F [UNA 's] ] [A [UNA plenty] [E [S [R [UNA of] ] [UNA parking] ] ] ] ] ... [L [UNA and] ] [+ [A [UNA I] ] [F [UNA 've] ] [T [UNA never] ] [F had] [F [UNA an] ] [UNA issue] [A [R [UNA with] ] [E [UNA audience] ] [UNA|A|P [UNA members] ] [E [+ [A\* members] [R [UNA who] ] [F [UNA wo] ] [D [UNA n't] ] [UNA stop] [^ [**+(COORD)** [+ [UNA talking] ] [L [UNA or] ] [+ [UNA answering] [A [E [A|S [UNA their] ] [A\* cellphones] ] [UNA cellphones] ] ] ] ] ] ] ] ] ] [U ,] ... [U .]

## Step 5: Decide `S` or `P` for remaining `+` scenes

Copula *be* and stative *have* are `S`; other verbs, as well as nouns tagged as ACT, PHENOMENON, PROCESS, or EVENT, are `P`.

> [S(COORD) [S [UNA There] [F [UNA 's] ] [A [UNA plenty] [E [S [R [UNA of] ] [UNA parking] ] ] ] ] ... [L [UNA and] ] [**P** [A [UNA I] ] [F [UNA 've] ] [T [UNA never] ] [F had] [F [UNA an] ] [UNA issue] [A [R [UNA with] ] [E [UNA audience] ] [UNA|A|P [UNA members] ] [E [**P** [A\* members] [R [UNA who] ] [F [UNA wo] ] [D [UNA n't] ] [UNA stop]
[^ [+(COORD) [**P** [UNA talking] ] [L [UNA or] ] [**P** [UNA answering] [A [E [A|S [UNA their] ] [A\* cellphones] ] [UNA cellphones] ] ] ] ]
] ] ] ] ] [U ,] ... [U .]

## Step 6.1: Restructure for secondary verbs

> [S(COORD) [S [UNA There] [F [UNA 's] ] [A [UNA plenty] [E [S [R [UNA of] ] [UNA parking] ] ] ] ] ... [L [UNA and] ] [P [A [UNA I] ] [F [UNA 've] ] [T [UNA never] ] [F had] [F [UNA an] ] [UNA issue] [A [R [UNA with] ] [E [UNA audience] ] [UNA|A|P [UNA members] ] [E [P [A\* members] [R [UNA who] ] [F [UNA wo] ] [D [UNA n't] ] [**D** [UNA stop] ]
[**+(COORD)** [P [UNA talking] ] [L [UNA or] ] [P [UNA answering] [A [E [A|S [UNA their] ] [A\* cellphones] ] [UNA cellphones] ] ] ]
] ] ] ] ] [U ,] ... [U .]

## Step 6.2: Articulation—marking lexical heads of units as `C`, `P`, or `S`, and renaming scene units as `H` where necessary; determination of `C` involves "X of Y" constructions involving quantities/Species

> [S(COORD) [**H(S)** [S [UNA There] ] [F [UNA 's] ] [A [**Q** [UNA plenty] ] [E [**H(S)** [R [UNA of] ] [**S** [UNA parking] ] ] ] ] ] ... [L [UNA and] ] [**H(P)** [A [UNA I] ] [F [UNA 've] ] [T [UNA never] ] [F had] [F [UNA an] ] [**P** [UNA issue] ] [A [R [UNA with] ] [E [UNA audience] ] [**H(A|P)|A|P** [**A|P** [UNA members] ] ] [E [**H** [A\* members] [R [UNA who] ] [F [UNA wo] ] [D [UNA n't] ] [D [UNA stop] ] [+(COORD) [**H(P)** [P [UNA talking] ] ] [L [UNA or] ] [**H(P)** [P [UNA answering] ] [A [E [**H(A|S)|S** [A|S [UNA their] ] ] [A\* cellphones] ] [**C** [UNA cellphones] ] ] ] ] ] ] ] ] ] [U ,] ... [U .]

TODO: BUG: The "of parking" PP unit should be C, not E? Or should Q not be combined with a scene-evoker?

## Steps 7+: Cleanup

Remove temporary decorations on `H` units from articulation; move `U` units for punctuation to more convenient attachment points; convert remaining `-` and `+` labels; wrap stray `P` and `S` units with `H` scenes; remove `UNA` and other temporary designations in the graph

> [S(COORD) [**H** [S [UNA There] ] [F [UNA 's] ] [A [Q [UNA plenty] ] [E [**H** [R [UNA of] ] [S [UNA parking] ] ] ] ] ] ... [L [UNA and] ] [**H** [A [UNA I] ] [F [UNA 've] ] [T [UNA never] ] [F had] [F [UNA an] ] [P [UNA issue] ] [A [R [UNA with] ] [E [UNA audience] ] [**C** [A|P [UNA members] ] ] [E [H [A\* members] [R [UNA who] ] [F [UNA wo] ] [D [UNA n't] ] [D [UNA stop] ] [+(COORD) [**H** [P [UNA talking] ] ] [L [UNA or] ] [**H** [P [UNA answering] ] [A [E [**C** [A|S [UNA their] ] ] [A\* cellphones] ] [C [UNA cellphones] ] ] ] ] ] ] ] ] ] [U ,] ... [U .]

> [S(COORD) [H [S [UNA There] ] [F [UNA 's] ] [A [Q [UNA plenty] ] [E [H [R [UNA of] ] [S [UNA parking] ] ] ] ] ] **[U ,]** [L [UNA and] ] [H [A [UNA I] ] [F [UNA 've] ] [T [UNA never] ] [F had] [F [UNA an] ] [P [UNA issue] ] [A [R [UNA with] ] [E [UNA audience] ] [C [A|P [UNA members] ] ] [E [H [A\* members] [R [UNA who] ] [F [UNA wo] ] [D [UNA n't] ] [D [UNA stop] ] [+(COORD) [H [P [UNA talking] ] ] [L [UNA or] ] [H [P [UNA answering] ] [A [E [C [A|S [UNA their] ] ] [A\* cellphones] ] [C [UNA cellphones] ] ] ] ] ] ] ] ] **[U .]** ]


> [**H** [H [S [UNA There] ] [F [UNA 's] ] [A [Q [UNA plenty] ] [E [H [R [UNA of] ] [S [UNA parking] ] ] ] ] ] [U ,] [L [UNA and] ] [H [A [UNA I] ] [F [UNA 've] ] [T [UNA never] ] [F had] [F [UNA an] ] [P [UNA issue] ] [A [R [UNA with] ] [E [UNA audience] ] [C [A|P [UNA members] ] ] [E [H [A\* members] [R [UNA who] ] [F [UNA wo] ] [D [UNA n't] ] [D [UNA stop] ] [**H** [H [P [UNA talking] ] ] [L [UNA or] ] [H [P [UNA answering] ] [A [E [C [A|S [UNA their] ] ] [A\* cellphones] ] [C [UNA cellphones] ] ] ] ] ] ] ] ] [U .] ]

> [H [H [S There] [F 's] [A [Q plenty] [E [R of] [S parking] ] ] ] [U ,] [L and] [H [A I] [F 've] [T never] [F had] [F an] [P issue] [A [R with] [E audience] [C [A|P members] ] [E [A\* members] [R who] [F wo] [D n't] [D stop] [H [H [P talking] ] [L or] [H [P answering] [A [E [C [A|S their] ] [A\* cellphones] ] [C cellphones] ] ] ] ] ] ] [U .] ]

## Gold for reference

> [H [**F** There] [F 's] [**D** plenty] [**P** [R of] [C parking] ] ] [U ,] [L and] [H [A I] [F 've] [**D|T** never] [F had] [P [F an] [C issue] ] [A [R with] [C [**A** audience] [**P** members] ] [E [R who] [**H** [A* members] [F wo] [D n't] [D stop] [P talking] ] [L or] [**H** [A* members] [P answering] [A [E [S|A their] [A* cellphones] ] [C cellphones] [U .] ] ] ] ] ]
