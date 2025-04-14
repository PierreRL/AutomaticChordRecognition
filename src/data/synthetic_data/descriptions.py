import random

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