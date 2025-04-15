import random

def generate_description():
    """
    Generate a random description for a music track.
    The description includes genre, adjectives, textures, rhythms, and instruments.
    """
    descriptions = [
        "A laid-back blues shuffle with a relaxed tempo, warm guitar tones, and a comfortable groove, perfect for a slow dance or a night in. Instruments: electric guitar, bass, drums, voices.",
        "A smooth acid jazz track with a laid-back groove, silky electric piano, and a cool bass, providing a modern take on jazz. Instruments: electric piano, bass, drums, backing vocals.",
        "A classic rock n' roll tune with catchy guitar riffs, driving drums, and a pulsating bass line, reminiscent of the golden era of rock. Instruments: electric guitar, bass, drums, backing vocals.",
        "A moody blues number with soulful guitar licks, mellow rhythms, and a walking bass line that sets a contemplative mood. Instruments: electric guitar, upright bass, drums, backing vocals.",
        "A funky acid jazz jam with syncopated rhythms, bright keyboard stabs, and a groove-heavy bassline. Instruments: electric piano, bass, congas, drums, backing vocals.",
        "A high-energy rock anthem with overdriven guitars, punchy snare hits, and a thunderous bass. Instruments: electric guitar, rhythm guitar, bass, drums, backing vocals.",
        "A vintage blues piece with expressive bends on the guitar, a slow swinging beat, and a deep bass foundation. Instruments: electric guitar, bass, drums, harmonica, backing vocals.",
        "A jazzy acid jazz groove with lush chord progressions, smooth keyboard lines, and laid-back percussion. Instruments: electric piano, bass, hand percussion, drums.",
        "A garage rock throwback with raw guitar tones, clattering drums, and a driving bass line. Instruments: electric guitar, bass, drums, organ, backing vocals.",
        "A heartfelt blues ballad with clean guitar picking, gentle drum brushes, and a round, soft bass. Instruments: electric guitar, bass, brushed drums, organ, backing vocals.",
        "A chill acid jazz vibe with mellow keyboard harmonies, minimalistic beats, and a flowing bassline. Instruments: electric piano, synth bass, drums, backing vocals.",
        "A fiery rock track with soaring guitar solos, hard-hitting drums, and a relentless groove. Instruments: electric guitar, bass, drums, tambourine, backing vocals.",
        "A traditional blues jam with call-and-response guitar phrases, steady backbeat, and a soulful low-end. Instruments: electric guitar, bass, drums, rhythm guitar, backing vocals.",
        "A stylish acid jazz instrumental with shimmering electric piano, tight bass lines, and crisp drumming. Instruments: electric piano, synth pads, bass, drums, backing vocals.",
        "A riff-heavy rock tune with crunchy guitars, steady bass, and energetic rhythms driving it forward. Instruments: electric guitar, bass, drums, organ, backing vocals."
    ]

    return random.choice(descriptions)


"""
# Musical elements to mix and match
genres = {
    "blues": ["laid-back", "gritty", "soulful", "mournful", "uplifting"],
    "jazz": ["smooth", "lively", "modern", "swinging", "sophisticated"],
    "rock": ["classic", "nostalgic", "grungy", "energetic", "heavy"],
    "funk": ["groovy", "tight", "explosive", "vibrant", "syncopated"],
    "metal": ["brutal", "ferocious", "epic", "dark", "relentless"],
    "pop": ["catchy", "radiant", "polished", "upbeat", "danceable"],
    "reggae": ["chilled", "rootsy", "sun-soaked", "dub-infused", "laid-back"],
    "hiphop": ["hard-hitting", "jazzy", "minimal", "old-school", "trap-style"],
    "electronic": ["ambient", "pulsating", "glitchy", "cinematic", "futuristic"]
}

# Template pools
moods = [
    "with a {adj1} vibe, {texture1}, and {rhythm1}",
    "featuring {texture2}, {element1}, and {adj2} rhythms",
    "built around {element2} and a {adj3} groove",
    "that blends {adj4} instrumentation and {rhythm2}",
    "highlighting {element3}, {element4}, and {adj5} percussion"
]

# Phrasing and adjectives
adjectives = ["laid-back", "gritty", "lush", "uplifting", "soulful", "syncopated", "minimal", "cinematic", "complex", "raw", "introspective", "dreamlike", "infectious"]
textures = ["warm guitar tones", "silky synths", "crisp drum loops", "emotive strings", "vintage samples", "swirling pads", "gritty distortion", "shimmering keys"]
rhythms = ["a bouncing beat", "tight grooves", "a relaxed tempo", "double-time percussion", "head-nodding rhythms", "a hypnotic pulse"]

# Instrument pool (will be sampled per genre)
instrument_pool = [
    "electric guitar", "acoustic guitar", "bass", "bass guitar", "drums", "piano", "upright bass",
    "electric piano", "synth", "organ", "trumpet", "saxophone", "violin", "cello", "flute",
    "harpsichord", "drum machine", "samples", "electronic effects", "banjo", "mandolin", "clavinet"
]

def generate_description():
    # Select genre and adjectives
    genre = random.choice(list(genres.keys()))
    genre_adj = random.choice(genres[genre])
    
    # Build description using templates
    mood_template = random.choice(moods)
    description_body = mood_template.format(
        adj1=random.choice(adjectives),
        adj2=random.choice(adjectives),
        adj3=random.choice(adjectives),
        adj4=random.choice(adjectives),
        adj5=random.choice(adjectives),
        texture1=random.choice(textures),
        texture2=random.choice(textures),
        rhythm1=random.choice(rhythms),
        rhythm2=random.choice(rhythms),
        element1=random.choice(textures),
        element2=random.choice(textures),
        element3=random.choice(textures),
        element4=random.choice(textures)
    )

    # Select 3–5 unique instruments
    instruments = random.sample(instrument_pool, random.randint(3, 5))
    instruments_str = "Instruments: " + ", ".join(instruments) + "."

    return f"A {genre_adj} {genre} track {description_body}. {instruments_str}"
"""