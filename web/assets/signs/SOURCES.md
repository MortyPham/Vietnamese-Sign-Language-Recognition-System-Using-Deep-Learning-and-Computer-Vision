# Sentence demonstration videos

These public videos are embedded using YouTube’s player. No source video is copied or redistributed. The demonstrations are learning references; they are not confirmed recordings from the model’s training set. Regional signing may differ.

Verified on 5 October 2026:

| Model label | Source | Video |
| --- | --- | --- |
| Xin chào. | [TokyoLife’s official lesson page](https://tokyolife.vn/blog/ngoi-nha-thien-than/cung-hoc-ngon-ngu-ky-hieu-voi-thien-than-tokyolife), lesson 1. Includes Xin chào, Cảm ơn and Xin lỗi. | [TokyoLife Channel](https://www.youtube.com/watch?v=4avGnxaLmb0) |
| Cảm ơn. | [Hanoi Sign Language Talking Dictionary, entry 104](https://talkingdictionary.swarthmore.edu/vietnamese_sign/?entry=104), Swarthmore College. Signer: Isaac Trương. | [Demonstration](https://www.youtube.com/watch?v=E5I_GHsdXAY) |
| Bạn tên là gì? | [Hanoi Sign Language Talking Dictionary, entry 109](https://talkingdictionary.swarthmore.edu/vietnamese_sign/?entry=109), Swarthmore College. Signer: Isaac Trương. | [Demonstration](https://www.youtube.com/watch?v=7kHmw59FUxI) |

The [original sentence dataset repository](https://github.com/khooinguyeen/Vietnamese-Sign-Language-Translation) publishes extracted landmark arrays rather than labeled recordings for all 60 supported phrases. Unverified matches are omitted from the catalog; the UI shows an unavailable state for them.

## Adding a verified demonstration

Add a record in `sentence-videos.json` with an exact model label, source attribution and either `provider: "youtube"` with its 11-character `video_id`, or `provider: "local"` with an owned/licensed MP4 under `/assets/signs/`. Local URLs must use lowercase letters, digits, slashes and hyphens. Optional YouTube `start` and `end` values are whole seconds. Only add timestamps checked against the actual video.

The shared interface loads the catalog independently of recognition, lists available videos first, and removes the player on close or navigation. Selecting a demonstration does not constrain or influence the model’s prediction.
