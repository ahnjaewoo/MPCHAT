## Next Response Prediction

**mpchat_nrp.json** \
Each split (train/val/test) contains a list of dialogues. \
A dialogue has the following structure:

```
[
    {
        ## Example (not real)
        "subreddit": "itookapicture",
        "messages": [
            "itap of my cat",
            "omg it is so cute! great shot!",
            "she is a model, she takes the pose for me."
        ],
        "message_ids": [
            "ab3d5f", ## post (or comment) id
            "gaq65vy",
            "gaquc1k"
        ],
        "main_author": "johndoe",
        "authors": [
            "johndoe",
            "mickeymouse",
            "johndoe"
        ]
        "created_utcs": [
            1604117284.0,
            1604173517.0,
            1604188317.0,
        ],
        "has_image": true,
        "all_personas": [
            {
                "id": "iawlby",
                "subreddit": "barista",
                "url" "https://i.redd.it/ghifjk.jpg",
                "score": 5, ## upvotes - downvotes
                "author": "johndoe",
                "created_utcs": 1597599418.0,
                "permalink": "/r/barista/comments/iawlby/my_dripper_art"
                "title": "my dripper art pieces that i made into stickers recently!",
                "direct_url": "https://i.redd.it/ghifjk.jpg",
                "file_name": "iawlby_ghifjk.jpg" ## {post_id}_{direct_url.split('/')[-1]}
            },
            ...
        ],
        "grounded_personas": [
            [],
            [],
            [
                {
                    "id": "jg9ml1",
                    "subreddit": "cats",
                    "url": "https://i.redd.it/0qBk1ej.jpg",
                    "score": 238,
                    "author": "johndoe",
                    "created_utc": 1597549415,
                    "permalink": "/r/cats/comments/jg9ml1/from_feline_to_fashion/",
                    "title": "from feline to fashion: my journey training a cat supermodel",
                    "direct_url": "https://i.redd.it/0qBk1ej.jpg",
                    "file_name": "jg9ml1_0qBk1ej.jpg",
                    "label_overall": "(strong) E",
                    #################################
                    ## (strong) E: response strongly entailed by (=grounded on) the persona element - entailment score = 3/3
                    ## E: response entailed by the persona element - entailment score = 2/3
                    ## I: response irrelevant to the persona element - entailment score = 1/3
                    ## (strong) I: response strongly irrelevant to persona element - entailment score = 0/3
                    #################################
                    "label_per_worker": [ ## three workers labeled it as entailed (=1)
                        [
                            "A2NSS746CFCT4N",
                            1
                        ],
                        [
                            "AKQAI78JTXXC8",
                            1,
                        ],
                        [
                            "ANX6Q4NMZL8EL",
                            1
                        ],
                }
            ]
        ],
        "ungrounded_personas": [
            [],
            [],
            []
        ],
        "direct_url": "https://i.redd.it/abcdef.jpg",
        "file_name": "ab3d5f_abcdef.jpg" ## {post_id}_{direct_url.split('/')[-1]}
        "candidate_personas": [ ## max 5 elements among 'all_personas'
            {
                "id": "iawlby",
                "subreddit": "barista",
                "url" "https://i.redd.it/ghifjk.jpg",
                "score": 5,
                "author": "johndoe",
                "created_utcs": 1597599418.0,
                "permalink": "/r/barista/comments/iawlby/my_dripper_art"
                "title": "my dripper art pieces that i made into stickers recently!",
                "direct_url": "https://i.redd.it/ghifjk.jpg",
                "file_name": "iawlby_ghifjk.jpg"
            },
            {
                "id": "jg9ml1",
                "subreddit": "cats",
                "url": "https://i.redd.it/0qBk1ej.jpg",
                "score": 238,
                "author": "johndoe",
                "created_utc": 1597549415,
                "permalink": "/r/cats/comments/jg9ml1/from_feline_to_fashion/",
                "title": "from feline to fashion: my journey training a cat supermodel",
                "direct_url": "https://i.redd.it/0qBk1ej.jpg",
                "file_name": "jg9ml1_0qBk1ej.jpg",
            },
            ...
        ],
        "nrp_candidate_responses": [ ## only in val and test set
            [
                "itap of my cat",
                (99 candidate responses)
                ...
            ],
            [],
            [
                "she is a model, she takes the pose for me."
                (99 candidate responses)
                ...
            ]
        ]
    },
    ...
]
```

Please see below for a description of each attribute in the dataset:

attribute | type | description
--- | ---  | ---
`subreddit` | str | subreddit of post
`messages` | list of str | dialogue between multiple authors
`message_ids` | list of str | post (or comment) id of each utterance
`main_author` | str | main author with multimodal persona info
`authors` | list of str | author info of each utterance
`created_utcs` | str | UTC epoch when post (or comment) was submitted
`has_image` | bool | whether post has image or not
`direct_url` | str | direct url of post image
`file_name` | str | saved file name of post image (format: {post_id}_{direct_url.split('/')[-1]})
`all_personas` | list of dict | main author's all persona elements
`grounded_personas` | list of list of dict | grounding persona elements of each utterance, only provided in main author's turn (labeled by workers)
`ungrounded_personas` | list of list of dict | un-grounding persona elements of each utterance, only provided in main author's turn (labeled by workers)
`candidate_personas` | list of dict | main author's candidate (max 5) persona elements
`nrp_candidate_responses` | list of list of str | 100 candidate respones, only provided in main author's turn

## Grounding Persona Prediction

**mpchat_gpp.json**  \
A dialogue has the following structure:

```
[
    {
        "subreddit": "itookapicture",
        "messages": [
            "itap of my cat",
            "omg it is so cute! great shot!",
            "she is a model, she takes the pose for me."
        ],
        "message_ids": [
            "ab3d5f",
            "gaq65vy",
            "gaquc1k"
        ],
        "main_author": "johndoe",
        "authors": [
            "johndoe",
            "mickeymouse",
            "johndoe"
        ]
        "created_utcs": [
            1604117284.0,
            1604173517.0,
            1604188317.0,
        ],
        "has_image": true,
        "all_personas": [
            ...
        ],
        "direct_url": "https://i.redd.it/abcdef.jpg",
        "file_name": "ab3d5f_abcdef.jpg"
        "gpp_grounded_persona": [ ## the persona element grounding on the response for each turn
            null,
            null,
            {
                "id": "jg9ml1",
                "subreddit": "cats",
                "url": "https://i.redd.it/0qBk1ej.jpg",
                "score": 238,
                "author": "johndoe",
                "created_utc": 1597549415,
                "permalink": "/r/cats/comments/jg9ml1/from_feline_to_fashion/",
                "title": "from feline to fashion: my journey training a cat supermodel",
                "direct_url": "https://i.redd.it/0qBk1ej.jpg",
                "file_name": "jg9ml1_0qBk1ej.jpg",
            }
        ],
        "gpp_candidate_personas": [ ## max 4 persona elements for each turn
            [],
            [],
            [
                {
                    "id": "iawlby",
                    "subreddit": "barista",
                    "url" "https://i.redd.it/ghifjk.jpg",
                    "score": 5,
                    "author": "johndoe",
                    "created_utcs": 1597599418.0,
                    "permalink": "/r/barista/comments/iawlby/my_dripper_art"
                    "title": "my dripper art pieces that i made into stickers recently!",
                    "direct_url": "https://i.redd.it/ghifjk.jpg",
                    "file_name": "iawlby_ghifjk.jpg"
                },
                ...
              ]
        ],
        "gpp_candidate_authors_candidate_personas": [ ## only in val and test set
            [],
            [],
            [
                {
                    "id": "jg9ml1",
                    ...
                }
                (99 candidate persona elements)
                ...
            ]
        ]
    },
    ...
]
```

Please see below for a description of each attribute in the dataset:

attribute | type | description
--- | ---  | ---
`subreddit` | str | subreddit of post
`messages` | list of str | dialogue between multiple authors
`message_ids` | list of str | post (or comment) id of each utterance
`main_author` | str | main author with multimodal persona info
`authors` | list of str | author info of each utterance
`created_utcs` | str | UTC epoch when post (or comment) was submitted
`has_image` | bool | whether post has image or not
`direct_url` | str | direct url of post image
`file_name` | str | saved file name of post image (format: {post_id}_{direct_url.split('/')[-1]})
`all_personas` | list of dict | main author's all persona elements
`gpp_grounded_persona` | list of dict | grounding persona element of each utterance, only provided in main author's turn
`gpp_candidate_personas` | list of list of dict | main author's candidate (max 4) persona elements, only provided in main author's turn and only if `gpp_grounded_persona` exists in the turn
`gpp_candidate_authors_candidate_personas` | list of list of dict | 100 candidate persona elements, only provided in main author's turn and only if `gpp_grounded_persona` exists in the turn

## Speaker Identification

**mpchat_si.json**  \
A dialogue has the following structure:

```
[
    {
        "subreddit": "itookapicture",
        "messages": [
            "itap of my cat",
            "omg it is so cute! great shot!",
            "she is a model, she takes the pose for me."
        ],
        "message_ids": [
            "ab3d5f",
            "gaq65vy",
            "gaquc1k"
        ],
        "main_author": "johndoe",
        "authors": [
            "johndoe",
            "mickeymouse",
            "johndoe"
        ]
        "created_utcs": [
            1604117284.0,
            1604173517.0,
            1604188317.0,
        ],
        "has_image": true,
        "all_personas": [
            {
                "id": "iawlby",
                "subreddit": "barista",
                "url" "https://i.redd.it/ghifjk.jpg",
                "score": 5, ## upvotes - downvotes
                "author": "johndoe",
                "created_utcs": 1597599418.0,
                "permalink": "/r/barista/comments/iawlby/my_dripper_art"
                "title": "my dripper art pieces that i made into stickers recently!",
                "direct_url": "https://i.redd.it/ghifjk.jpg",
                "file_name": "iawlby_ghifjk.jpg"
            },
            ...
        ],
        "direct_url": "https://i.redd.it/abcdef.jpg",
        "file_name": "ab3d5f_abcdef.jpg"
        "si_main_author_candidate_personas": [ ## max 5 elements among 'all_personas'
            {
                "id": "iawlby",
                "subreddit": "barista",
                "url" "https://i.redd.it/ghifjk.jpg",
                "score": 5,
                "author": "johndoe",
                "created_utcs": 1597599418.0,
                "permalink": "/r/barista/comments/iawlby/my_dripper_art"
                "title": "my dripper art pieces that i made into stickers recently!",
                "direct_url": "https://i.redd.it/ghifjk.jpg",
                "file_name": "iawlby_ghifjk.jpg"
            },
            {
                "id": "jg9ml1",
                "subreddit": "cats",
                "url": "https://i.redd.it/0qBk1ej.jpg",
                "score": 238,
                "author": "johndoe",
                "created_utc": 1597549415,
                "permalink": "/r/cats/comments/jg9ml1/from_feline_to_fashion/",
                "title": "from feline to fashion: my journey training a cat supermodel",
                "direct_url": "https://i.redd.it/0qBk1ej.jpg",
                "file_name": "jg9ml1_0qBk1ej.jpg",
            },
            ...
        ],
        "si_candidate_authors_candidate_personas": [ ## only in val and test set
            [
                {
                    "id": "jg9ml1",
                    ...
                },
                {
                    "id": "iawlby",
                    ...
                },
                ...
            ],
            (99 candidate authors' candidate personas)
        ]
    },
    ...
]
```

Please see below for a description of each attribute in the dataset:

attribute | type | description
--- | ---  | ---
`subreddit` | str | subreddit of post
`messages` | list of str | dialogue between multiple authors
`message_ids` | list of str | post (or comment) id of each utterance
`main_author` | str | main author with multimodal persona info
`authors` | list of str | author info of each utterance
`created_utcs` | str | UTC epoch when post (or comment) was submitted
`has_image` | bool | whether post has image or not
`direct_url` | str | direct url of post image
`file_name` | str | saved file name of post image (format: {post_id}_{direct_url.split('/')[-1]})
`all_personas` | list of dict | main author's all persona elements
`si_main_author_candidate_personas` | list of dict | main author's candidate (max 5) persona elements
`si_candidate_authors_candidate_personas` | list of list of dict | 100 candidate authors' persona elements
