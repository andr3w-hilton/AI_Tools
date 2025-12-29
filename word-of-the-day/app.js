// Word database - curated collection of interesting words for one full year
// December 26, 2025 - December 31, 2026
const wordDatabase = [
    // Starting Friday, December 26, 2025
    { word: "Halcyon", pronunciation: "HAL-see-on", pos: "adjective", definition: "Denoting a period of time in the past that was idyllically happy and peaceful.", example: "They often reminisced about the halcyon days of their youth." },

    // Week 1: Dec 29-31, 2025
    { word: "Serendipity", pronunciation: "ser-en-DIP-ih-tee", pos: "noun", definition: "The occurrence of events by chance in a happy or beneficial way.", example: "Meeting her best friend was pure serendipity—they sat next to each other on a plane." },
    { word: "Ephemeral", pronunciation: "eh-FEM-er-ul", pos: "adjective", definition: "Lasting for a very short time; transient and fleeting.", example: "The ephemeral beauty of cherry blossoms reminds us to appreciate each moment." },
    { word: "Luminous", pronunciation: "LOO-min-us", pos: "adjective", definition: "Bright or shining, especially in the dark; full of or shedding light.", example: "Her luminous smile could brighten even the darkest days." },

    // Week 2: Jan 2026
    { word: "Mellifluous", pronunciation: "meh-LIF-loo-us", pos: "adjective", definition: "Sweet or musical; pleasant to hear.", example: "The singer's mellifluous voice filled the concert hall with warmth." },
    { word: "Effervescent", pronunciation: "ef-er-VES-ent", pos: "adjective", definition: "Vivacious and enthusiastic; giving off bubbles; fizzy.", example: "Her effervescent personality made everyone around her feel energized." },
    { word: "Petrichor", pronunciation: "PET-ri-kor", pos: "noun", definition: "The pleasant, earthy smell after rain.", example: "The petrichor rising from the garden after the storm was incredibly refreshing." },
    { word: "Eloquent", pronunciation: "EL-oh-kwent", pos: "adjective", definition: "Fluent or persuasive in speaking or writing.", example: "Her eloquent speech moved the audience to tears." },
    { word: "Ethereal", pronunciation: "eh-THEER-ee-ul", pos: "adjective", definition: "Extremely delicate and light; seeming too perfect for this world.", example: "The ballet dancer moved with an ethereal grace that mesmerized the audience." },

    // Week 3
    { word: "Incandescent", pronunciation: "in-kan-DES-ent", pos: "adjective", definition: "Emitting light as a result of being heated; passionate or brilliant.", example: "She gave an incandescent performance that earned her a standing ovation." },
    { word: "Ebullient", pronunciation: "eh-BULL-yent", pos: "adjective", definition: "Cheerful and full of energy; enthusiastic.", example: "The children were ebullient on Christmas morning, racing to open their presents." },
    { word: "Sanguine", pronunciation: "SANG-gwin", pos: "adjective", definition: "Optimistic or positive, especially in a difficult situation.", example: "Despite the setbacks, she remained sanguine about the project's success." },
    { word: "Quintessential", pronunciation: "kwin-teh-SEN-shul", pos: "adjective", definition: "Representing the most perfect or typical example of something.", example: "The cottage, with its thatched roof and garden, was the quintessential English home." },
    { word: "Ineffable", pronunciation: "in-EF-uh-bul", pos: "adjective", definition: "Too great or extreme to be expressed in words; indescribable.", example: "The ineffable joy of becoming a parent left them speechless." },

    // Week 4
    { word: "Resplendent", pronunciation: "reh-SPLEN-dent", pos: "adjective", definition: "Attractive and impressive through being richly colorful or sumptuous.", example: "The peacock's resplendent plumage shimmered in the sunlight." },
    { word: "Verdant", pronunciation: "VER-dent", pos: "adjective", definition: "Green with grass or other rich vegetation.", example: "The verdant hills of Ireland stretched as far as the eye could see." },
    { word: "Sonorous", pronunciation: "suh-NOR-us", pos: "adjective", definition: "Capable of producing a deep or ringing sound; impressive in style.", example: "The actor's sonorous voice commanded attention whenever he spoke." },
    { word: "Benevolent", pronunciation: "beh-NEV-oh-lent", pos: "adjective", definition: "Well-meaning and kindly; serving a charitable purpose.", example: "The benevolent stranger helped them carry their groceries home." },
    { word: "Vivacious", pronunciation: "vih-VAY-shus", pos: "adjective", definition: "Attractively lively and animated, especially of a woman.", example: "Her vivacious nature made her the life of every party." },

    // Week 5
    { word: "Redolent", pronunciation: "RED-oh-lent", pos: "adjective", definition: "Strongly reminiscent or suggestive of something; fragrant or sweet-smelling.", example: "The kitchen was redolent with the aroma of freshly baked bread." },
    { word: "Perspicacious", pronunciation: "per-spih-KAY-shus", pos: "adjective", definition: "Having a ready insight into and understanding of things; perceptive.", example: "Her perspicacious observations helped solve the mystery." },
    { word: "Magnanimous", pronunciation: "mag-NAN-ih-mus", pos: "adjective", definition: "Generous or forgiving, especially toward a rival or less powerful person.", example: "The champion was magnanimous in victory, praising her opponent's skill." },
    { word: "Loquacious", pronunciation: "loh-KWAY-shus", pos: "adjective", definition: "Tending to talk a great deal; talkative.", example: "The loquacious taxi driver shared stories throughout the entire journey." },
    { word: "Iridescent", pronunciation: "ear-ih-DES-ent", pos: "adjective", definition: "Showing luminous colors that seem to change when seen from different angles.", example: "The soap bubbles created an iridescent display in the sunlight." },

    // Week 6
    { word: "Euphonious", pronunciation: "yoo-FOH-nee-us", pos: "adjective", definition: "Pleasing to the ear; harmonious in sound.", example: "The euphonious melody of the harp filled the room." },
    { word: "Sublime", pronunciation: "suh-BLIME", pos: "adjective", definition: "Of such excellence, grandeur, or beauty as to inspire great admiration.", example: "The view from the mountain peak was absolutely sublime." },
    { word: "Sagacious", pronunciation: "suh-GAY-shus", pos: "adjective", definition: "Having or showing keen mental discernment and good judgment; wise.", example: "The sagacious advisor helped them avoid many costly mistakes." },
    { word: "Sumptuous", pronunciation: "SUMP-choo-us", pos: "adjective", definition: "Splendid and expensive-looking; magnificent.", example: "They enjoyed a sumptuous feast at the wedding reception." },
    { word: "Ameliorate", pronunciation: "uh-MEEL-yor-ate", pos: "verb", definition: "To make something bad or unsatisfactory better.", example: "The new policies helped ameliorate working conditions in the factory." },

    // Week 7
    { word: "Scintillating", pronunciation: "SIN-tih-lay-ting", pos: "adjective", definition: "Sparkling or shining brightly; brilliantly and excitingly clever or skillful.", example: "Her scintillating wit kept everyone entertained throughout dinner." },
    { word: "Ubiquitous", pronunciation: "yoo-BIK-wih-tus", pos: "adjective", definition: "Present, appearing, or found everywhere.", example: "Smartphones have become ubiquitous in modern society." },
    { word: "Cacophony", pronunciation: "kuh-KOF-uh-nee", pos: "noun", definition: "A harsh, discordant mixture of sounds.", example: "The cacophony of car horns and sirens filled the city streets." },
    { word: "Tenacity", pronunciation: "teh-NAS-ih-tee", pos: "noun", definition: "The quality of being determined and persistent.", example: "Her tenacity in pursuing her dreams eventually led to success." },
    { word: "Ineffable", pronunciation: "in-EF-uh-bul", pos: "adjective", definition: "Too great or extreme to be expressed in words.", example: "The ineffable beauty of the sunset left them speechless." },

    // Week 8
    { word: "Obfuscate", pronunciation: "OB-fuh-skayt", pos: "verb", definition: "To deliberately make something unclear or difficult to understand.", example: "The politician tried to obfuscate the issue with complex jargon." },
    { word: "Ephemeral", pronunciation: "eh-FEM-er-ul", pos: "adjective", definition: "Lasting for a very short time.", example: "The morning dew was ephemeral, disappearing with the sunrise." },
    { word: "Labyrinthine", pronunciation: "lab-ih-RIN-thine", pos: "adjective", definition: "Like a labyrinth; irregular and twisting.", example: "The labyrinthine streets of the old city confused even local residents." },
    { word: "Zenith", pronunciation: "ZEE-nith", pos: "noun", definition: "The highest point reached by a celestial or other object; the peak.", example: "The company reached its zenith in the 1990s before declining." },
    { word: "Mercurial", pronunciation: "mer-KYOOR-ee-ul", pos: "adjective", definition: "Subject to sudden or unpredictable changes of mood or mind.", example: "His mercurial temperament made him difficult to work with." },

    // Week 9
    { word: "Languorous", pronunciation: "LANG-ger-us", pos: "adjective", definition: "Characterized by a pleasant feeling of tiredness or inactivity.", example: "They spent a languorous afternoon lounging by the pool." },
    { word: "Beguile", pronunciation: "bih-GYLE", pos: "verb", definition: "To charm or enchant someone, often in a deceptive way.", example: "The con artist used charm to beguile unsuspecting victims." },
    { word: "Pristine", pronunciation: "PRIS-teen", pos: "adjective", definition: "In its original condition; unspoiled; clean and fresh.", example: "The pristine snow covered the mountain peaks." },
    { word: "Ephemeral", pronunciation: "eh-FEM-er-ul", pos: "adjective", definition: "Lasting for a very short time.", example: "Social media fame is often ephemeral and fleeting." },
    { word: "Resonate", pronunciation: "REZ-uh-nayt", pos: "verb", definition: "To evoke images, memories, and emotions; to relate harmoniously.", example: "Her words resonated deeply with the audience." },

    // Week 10
    { word: "Enigmatic", pronunciation: "en-ig-MAT-ik", pos: "adjective", definition: "Difficult to interpret or understand; mysterious.", example: "The Mona Lisa's enigmatic smile has puzzled viewers for centuries." },
    { word: "Plethora", pronunciation: "PLETH-or-uh", pos: "noun", definition: "An excessive amount of something.", example: "The buffet offered a plethora of delicious options." },
    { word: "Ameliorate", pronunciation: "uh-MEEL-yor-ate", pos: "verb", definition: "To make something better; to improve.", example: "The new medication helped ameliorate her symptoms." },
    { word: "Voracious", pronunciation: "vor-AY-shus", pos: "adjective", definition: "Having a very eager approach to an activity; wanting or devouring great quantities of food.", example: "She was a voracious reader, finishing several books each week." },
    { word: "Intrinsic", pronunciation: "in-TRIN-sik", pos: "adjective", definition: "Belonging naturally; essential.", example: "Curiosity is intrinsic to human nature." },

    // Week 11
    { word: "Nostalgia", pronunciation: "noss-TAL-juh", pos: "noun", definition: "A sentimental longing for the past.", example: "Looking at old photographs filled her with nostalgia." },
    { word: "Prodigious", pronunciation: "pruh-DIJ-us", pos: "adjective", definition: "Remarkably or impressively great in extent, size, or degree.", example: "Mozart displayed prodigious talent from a very young age." },
    { word: "Quintessence", pronunciation: "kwin-TESS-ens", pos: "noun", definition: "The most perfect or typical example of a quality or class.", example: "Paris is often considered the quintessence of romantic cities." },
    { word: "Vicarious", pronunciation: "vy-KAIR-ee-us", pos: "adjective", definition: "Experienced in the imagination through the feelings or actions of another person.", example: "She lived vicariously through her daughter's adventures abroad." },
    { word: "Fastidious", pronunciation: "fass-TID-ee-us", pos: "adjective", definition: "Very attentive to detail; meticulous.", example: "He was fastidious about keeping his workspace organized." },

    // Week 12
    { word: "Ostentatious", pronunciation: "oss-ten-TAY-shus", pos: "adjective", definition: "Characterized by vulgar or pretentious display; designed to impress.", example: "The mansion's ostentatious decor screamed nouveau riche." },
    { word: "Paradigm", pronunciation: "PAIR-uh-dime", pos: "noun", definition: "A typical example or pattern of something; a model.", example: "The discovery created a paradigm shift in scientific thinking." },
    { word: "Juxtapose", pronunciation: "JUCKS-tuh-poze", pos: "verb", definition: "To place things side by side for contrasting effect.", example: "The artist juxtaposed bright colors with dark shadows." },
    { word: "Nefarious", pronunciation: "neh-FAIR-ee-us", pos: "adjective", definition: "Wicked or criminal; villainous.", example: "The villain hatched a nefarious plot to take over the city." },
    { word: "Zephyr", pronunciation: "ZEF-er", pos: "noun", definition: "A soft gentle breeze.", example: "A light zephyr rustled through the trees on the warm afternoon." },

    // Week 13
    { word: "Quixotic", pronunciation: "kwik-SOT-ik", pos: "adjective", definition: "Exceedingly idealistic; unrealistic and impractical.", example: "His quixotic quest to end all poverty seemed admirable but impossible." },
    { word: "Languid", pronunciation: "LANG-gwid", pos: "adjective", definition: "Lacking vigor or vitality; slow and relaxed.", example: "They spent a languid summer afternoon by the lake." },
    { word: "Ephemeral", pronunciation: "eh-FEM-er-ul", pos: "adjective", definition: "Lasting for a very short time.", example: "The firefly's glow was ephemeral but magical." },
    { word: "Perennial", pronunciation: "puh-REN-ee-ul", pos: "adjective", definition: "Lasting or existing for a long time; enduring or continually recurring.", example: "World hunger remains a perennial problem." },
    { word: "Bucolic", pronunciation: "byoo-KOL-ik", pos: "adjective", definition: "Relating to the pleasant aspects of the countryside and country life.", example: "They retired to a bucolic village in the countryside." },

    // Week 14
    { word: "Ephemeral", pronunciation: "eh-FEM-er-ul", pos: "adjective", definition: "Lasting for a very short time.", example: "Fame on the internet is often ephemeral." },
    { word: "Gregarious", pronunciation: "greh-GAIR-ee-us", pos: "adjective", definition: "Fond of company; sociable.", example: "Her gregarious nature made her popular at parties." },
    { word: "Harbinger", pronunciation: "HAR-bin-jer", pos: "noun", definition: "A person or thing that announces or signals the approach of another.", example: "Robins are often seen as harbingers of spring." },
    { word: "Idyllic", pronunciation: "eye-DIL-ik", pos: "adjective", definition: "Extremely happy, peaceful, or picturesque.", example: "They spent an idyllic weekend at the beach cottage." },
    { word: "Kinetic", pronunciation: "kih-NET-ik", pos: "adjective", definition: "Relating to or resulting from motion.", example: "The sculpture's kinetic energy captivated museum visitors." },

    // Week 15
    { word: "Laconic", pronunciation: "luh-KON-ik", pos: "adjective", definition: "Using very few words; brief and concise.", example: "His laconic response of 'maybe' left everyone confused." },
    { word: "Myriad", pronunciation: "MEER-ee-ad", pos: "noun/adjective", definition: "A countless or extremely great number.", example: "The night sky displayed myriad stars." },
    { word: "Nebulous", pronunciation: "NEB-yoo-lus", pos: "adjective", definition: "Unclear, vague, or ill-defined.", example: "His plans for the future remained nebulous and uncertain." },
    { word: "Opulent", pronunciation: "OP-yoo-lent", pos: "adjective", definition: "Ostentatiously rich and luxurious.", example: "The opulent ballroom featured crystal chandeliers and gold trim." },
    { word: "Panacea", pronunciation: "pan-uh-SEE-uh", pos: "noun", definition: "A solution or remedy for all difficulties or diseases.", example: "Some people mistakenly view technology as a panacea for all societal problems." },

    // Week 16
    { word: "Querulous", pronunciation: "KWER-uh-lus", pos: "adjective", definition: "Complaining in a petulant or whining manner.", example: "The querulous customer demanded to speak to the manager." },
    { word: "Recalcitrant", pronunciation: "rih-KAL-sih-trent", pos: "adjective", definition: "Having an obstinately uncooperative attitude.", example: "The recalcitrant student refused to follow classroom rules." },
    { word: "Serendipitous", pronunciation: "ser-en-DIP-ih-tus", pos: "adjective", definition: "Occurring or discovered by chance in a happy or beneficial way.", example: "Their serendipitous meeting led to a lifelong friendship." },
    { word: "Taciturn", pronunciation: "TASS-ih-turn", pos: "adjective", definition: "Reserved or uncommunicative in speech; saying little.", example: "The taciturn detective rarely spoke unless necessary." },
    { word: "Ubiquitous", pronunciation: "yoo-BIK-wih-tus", pos: "adjective", definition: "Present, appearing, or found everywhere.", example: "Coffee shops have become ubiquitous in urban areas." },

    // Week 17
    { word: "Venerate", pronunciation: "VEN-er-ate", pos: "verb", definition: "To regard with great respect or reverence.", example: "The community venerated the elderly teacher for her decades of service." },
    { word: "Wistful", pronunciation: "WIST-ful", pos: "adjective", definition: "Having or showing a feeling of vague or regretful longing.", example: "She cast a wistful glance at the old house where she grew up." },
    { word: "Xenophile", pronunciation: "ZEN-oh-file", pos: "noun", definition: "A person attracted to foreign peoples, cultures, or customs.", example: "As a xenophile, she collected art and music from around the world." },
    { word: "Yearning", pronunciation: "YURN-ing", pos: "noun", definition: "A feeling of intense longing for something.", example: "He felt a deep yearning for his homeland." },
    { word: "Zealous", pronunciation: "ZEL-us", pos: "adjective", definition: "Having or showing zeal; fervent and enthusiastic.", example: "The zealous volunteers worked tirelessly on the campaign." },

    // Week 18
    { word: "Alacrity", pronunciation: "uh-LAK-rih-tee", pos: "noun", definition: "Brisk and cheerful readiness.", example: "She accepted the invitation with alacrity." },
    { word: "Bombastic", pronunciation: "bom-BASS-tik", pos: "adjective", definition: "High-sounding but with little meaning; inflated.", example: "The politician's bombastic speech was full of empty promises." },
    { word: "Capricious", pronunciation: "kuh-PRISH-us", pos: "adjective", definition: "Given to sudden and unaccountable changes of mood or behavior.", example: "The capricious weather made planning difficult." },
    { word: "Disparate", pronunciation: "DISS-puh-ret", pos: "adjective", definition: "Essentially different in kind; not allowing comparison.", example: "The team was formed from disparate individuals with different skills." },
    { word: "Eclectic", pronunciation: "eh-KLEK-tik", pos: "adjective", definition: "Deriving ideas, style, or taste from a broad and diverse range of sources.", example: "Her eclectic taste in music ranged from classical to hip-hop." },

    // Week 19
    { word: "Furtive", pronunciation: "FUR-tiv", pos: "adjective", definition: "Attempting to avoid notice or attention; secretive.", example: "He cast a furtive glance around the room before speaking." },
    { word: "Grandiose", pronunciation: "GRAN-dee-oze", pos: "adjective", definition: "Impressive or magnificent in appearance or style, especially pretentiously so.", example: "His grandiose plans for the project exceeded the budget." },
    { word: "Hackneyed", pronunciation: "HAK-need", pos: "adjective", definition: "Lacking significance through having been overused; unoriginal and trite.", example: "The movie relied on hackneyed plot devices." },
    { word: "Impetuous", pronunciation: "im-PECH-oo-us", pos: "adjective", definition: "Acting or done quickly and without thought or care.", example: "Her impetuous decision to quit her job surprised everyone." },
    { word: "Judicious", pronunciation: "joo-DISH-us", pos: "adjective", definition: "Having, showing, or done with good judgment or sense.", example: "The CEO made a judicious decision to expand gradually." },

    // Week 20
    { word: "Kaleidoscope", pronunciation: "kuh-LYE-duh-skope", pos: "noun", definition: "A constantly changing pattern or sequence of objects or elements.", example: "The city's culture is a kaleidoscope of different traditions." },
    { word: "Lugubrious", pronunciation: "loo-GOO-bree-us", pos: "adjective", definition: "Looking or sounding sad and dismal.", example: "The lugubrious music matched the somber mood of the funeral." },
    { word: "Mendacious", pronunciation: "men-DAY-shus", pos: "adjective", definition: "Not telling the truth; lying.", example: "The mendacious witness was caught in several contradictions." },
    { word: "Nonchalant", pronunciation: "non-shuh-LONT", pos: "adjective", definition: "Feeling or appearing casually calm and relaxed.", example: "She gave a nonchalant shrug when asked about the controversy." },
    { word: "Obsequious", pronunciation: "ob-SEE-kwee-us", pos: "adjective", definition: "Obedient or attentive to an excessive or servile degree.", example: "The obsequious waiter hovered constantly around the table." },

    // Week 21
    { word: "Paragon", pronunciation: "PAIR-uh-gon", pos: "noun", definition: "A person or thing regarded as a perfect example of a particular quality.", example: "She was considered a paragon of virtue in the community." },
    { word: "Quandary", pronunciation: "KWON-duh-ree", pos: "noun", definition: "A state of perplexity or uncertainty over what to do in a difficult situation.", example: "She found herself in a quandary about which job offer to accept." },
    { word: "Raucous", pronunciation: "RAW-kus", pos: "adjective", definition: "Making or constituting a disturbingly harsh and loud noise.", example: "The raucous laughter from the party disturbed the neighbors." },
    { word: "Sycophant", pronunciation: "SIK-uh-fant", pos: "noun", definition: "A person who acts obsequiously toward someone important in order to gain advantage.", example: "The office sycophant always agreed with the boss, no matter what." },
    { word: "Tangential", pronunciation: "tan-JEN-shul", pos: "adjective", definition: "Relating to or along a tangent; diverging from a previous course; only slightly connected.", example: "His tangential comments often derailed the meeting." },

    // Week 22
    { word: "Unctuous", pronunciation: "UNK-choo-us", pos: "adjective", definition: "Excessively or ingratiatingly flattering; oily.", example: "The salesman's unctuous manner made customers uncomfortable." },
    { word: "Verisimilitude", pronunciation: "ver-ih-sih-MIL-ih-tood", pos: "noun", definition: "The appearance of being true or real.", example: "The novel's verisimilitude made readers believe it was based on real events." },
    { word: "Winsome", pronunciation: "WIN-sum", pos: "adjective", definition: "Attractive or appealing in appearance or character.", example: "Her winsome smile won over everyone she met." },
    { word: "Xeric", pronunciation: "ZEER-ik", pos: "adjective", definition: "Of, relating to, or adapted to a dry environment.", example: "Cacti and succulents are xeric plants." },
    { word: "Yoke", pronunciation: "YOKE", pos: "noun/verb", definition: "A wooden crosspiece fastened over animals' necks; something that connects or binds.", example: "They worked to throw off the yoke of oppression." },

    // Week 23
    { word: "Abstruse", pronunciation: "ab-STROOSE", pos: "adjective", definition: "Difficult to understand; obscure.", example: "The professor's abstruse theories confused most students." },
    { word: "Belligerent", pronunciation: "buh-LIJ-er-ent", pos: "adjective", definition: "Hostile and aggressive.", example: "His belligerent attitude made negotiations difficult." },
    { word: "Cognizant", pronunciation: "KOG-nih-zent", pos: "adjective", definition: "Having knowledge or being aware of.", example: "She was cognizant of the risks involved in the venture." },
    { word: "Deleterious", pronunciation: "del-ih-TEER-ee-us", pos: "adjective", definition: "Causing harm or damage.", example: "Smoking has deleterious effects on health." },
    { word: "Ephemeral", pronunciation: "eh-FEM-er-ul", pos: "adjective", definition: "Lasting for a very short time.", example: "The popularity of internet memes is typically ephemeral." },

    // Week 24
    { word: "Facetious", pronunciation: "fuh-SEE-shus", pos: "adjective", definition: "Treating serious issues with deliberately inappropriate humor.", example: "His facetious comments during the meeting annoyed his colleagues." },
    { word: "Garrulous", pronunciation: "GAR-uh-lus", pos: "adjective", definition: "Excessively talkative, especially on trivial matters.", example: "The garrulous neighbor talked for hours about nothing important." },
    { word: "Hegemony", pronunciation: "heh-JEM-uh-nee", pos: "noun", definition: "Leadership or dominance, especially by one country or social group.", example: "The nation sought to establish hegemony over the region." },
    { word: "Iconoclast", pronunciation: "eye-KON-uh-klast", pos: "noun", definition: "A person who attacks cherished beliefs or institutions.", example: "As an iconoclast, she challenged traditional teaching methods." },
    { word: "Jocund", pronunciation: "JOK-und", pos: "adjective", definition: "Cheerful and lighthearted.", example: "The jocund atmosphere at the party lifted everyone's spirits." },

    // Week 25
    { word: "Lachrymose", pronunciation: "LAK-rih-moze", pos: "adjective", definition: "Tearful or given to weeping.", example: "The lachrymose farewell scene moved the audience to tears." },
    { word: "Malfeasance", pronunciation: "mal-FEE-zens", pos: "noun", definition: "Wrongdoing, especially by a public official.", example: "The mayor was accused of malfeasance in office." },
    { word: "Nascent", pronunciation: "NAY-sent", pos: "adjective", definition: "Just coming into existence and beginning to display signs of future potential.", example: "The nascent technology showed promise but needed development." },
    { word: "Obdurate", pronunciation: "OB-dur-it", pos: "adjective", definition: "Stubbornly refusing to change one's opinion or course of action.", example: "Despite pleas from his family, he remained obdurate in his decision." },
    { word: "Perfidious", pronunciation: "per-FID-ee-us", pos: "adjective", definition: "Deceitful and untrustworthy.", example: "The perfidious betrayal shocked everyone who trusted him." },

    // Week 26
    { word: "Quiescent", pronunciation: "kwee-ES-ent", pos: "adjective", definition: "In a state or period of inactivity or dormancy.", example: "The volcano had been quiescent for decades." },
    { word: "Recondite", pronunciation: "REK-un-dite", pos: "adjective", definition: "Little known; abstruse.", example: "The scholar specialized in recondite areas of medieval history." },
    { word: "Salubrious", pronunciation: "suh-LOO-bree-us", pos: "adjective", definition: "Health-giving; healthy.", example: "The salubrious mountain air helped his recovery." },
    { word: "Truculent", pronunciation: "TRUCK-yoo-lent", pos: "adjective", definition: "Eager or quick to argue or fight; aggressively defiant.", example: "The truculent customer refused to leave the store." },
    { word: "Umbrage", pronunciation: "UM-brij", pos: "noun", definition: "Offense or annoyance.", example: "She took umbrage at his dismissive tone." },

    // Week 27
    { word: "Vacillate", pronunciation: "VASS-ih-late", pos: "verb", definition: "To alternate or waver between different opinions or actions.", example: "He vacillated between accepting and rejecting the offer." },
    { word: "Wanton", pronunciation: "WON-tun", pos: "adjective", definition: "Deliberate and unprovoked; unrestrained.", example: "The vandals caused wanton destruction to the park." },
    { word: "Xenial", pronunciation: "ZEE-nee-ul", pos: "adjective", definition: "Relating to hospitality or relations between host and guest.", example: "The xenial traditions of the village welcomed all visitors warmly." },
    { word: "Yammer", pronunciation: "YAM-er", pos: "verb", definition: "To talk volubly and at length.", example: "The children yammered excitedly about their field trip." },
    { word: "Zeal", pronunciation: "ZEEL", pos: "noun", definition: "Great energy or enthusiasm in pursuit of a cause or objective.", example: "Her zeal for environmental protection inspired others to join the cause." },

    // Week 28
    { word: "Aphorism", pronunciation: "AF-or-iz-um", pos: "noun", definition: "A pithy observation that contains a general truth.", example: "The old aphorism 'actions speak louder than words' still rings true." },
    { word: "Bucolic", pronunciation: "byoo-KOL-ik", pos: "adjective", definition: "Relating to the pleasant aspects of the countryside.", example: "They enjoyed the bucolic scenery during their countryside retreat." },
    { word: "Circumlocution", pronunciation: "ser-kum-loh-KYOO-shun", pos: "noun", definition: "The use of many words where fewer would do; evasive speech.", example: "His circumlocution made it hard to understand his actual position." },
    { word: "Dilettante", pronunciation: "DIL-uh-tahnt", pos: "noun", definition: "A person who cultivates an area of interest without real commitment or knowledge.", example: "He was merely a dilettante, not a serious artist." },
    { word: "Equanimity", pronunciation: "ee-kwuh-NIM-ih-tee", pos: "noun", definition: "Mental calmness and evenness of temper, especially in difficult situations.", example: "She faced the crisis with remarkable equanimity." },

    // Week 29
    { word: "Flippant", pronunciation: "FLIP-ent", pos: "adjective", definition: "Not showing a serious or respectful attitude.", example: "His flippant remarks about the tragedy offended many people." },
    { word: "Glib", pronunciation: "GLIB", pos: "adjective", definition: "Fluent and voluble but insincere and shallow.", example: "The politician's glib answers failed to address the real issues." },
    { word: "Hubris", pronunciation: "HYOO-bris", pos: "noun", definition: "Excessive pride or self-confidence.", example: "His hubris led him to underestimate his competitors." },
    { word: "Insipid", pronunciation: "in-SIP-id", pos: "adjective", definition: "Lacking flavor; lacking vigor or interest.", example: "The insipid conversation bored everyone at the dinner party." },
    { word: "Jaded", pronunciation: "JAY-did", pos: "adjective", definition: "Tired, bored, or lacking enthusiasm, typically after having had too much of something.", example: "After years in the industry, she had become jaded and cynical." },

    // Week 30
    { word: "Kindle", pronunciation: "KIN-dul", pos: "verb", definition: "To light or set on fire; to arouse or inspire an emotion or feeling.", example: "The teacher's enthusiasm kindled a love of learning in her students." },
    { word: "Languish", pronunciation: "LANG-gwish", pos: "verb", definition: "To lose or lack vitality; to be forced to remain in an unpleasant place or situation.", example: "The reforms languished in committee for months." },
    { word: "Maudlin", pronunciation: "MAWD-lin", pos: "adjective", definition: "Self-pityingly or tearfully sentimental.", example: "After a few drinks, he became maudlin about his lost love." },
    { word: "Nuance", pronunciation: "NOO-ahnss", pos: "noun", definition: "A subtle difference in shade of meaning, appearance, or sound.", example: "The actor captured every nuance of the complex character." },
    { word: "Obtuse", pronunciation: "ob-TOOSE", pos: "adjective", definition: "Annoyingly insensitive or slow to understand.", example: "He was being deliberately obtuse about the obvious problem." },

    // Week 31
    { word: "Palliate", pronunciation: "PAL-ee-ate", pos: "verb", definition: "To make a disease or its symptoms less severe without removing the cause.", example: "The medication helped palliate the pain but didn't cure the condition." },
    { word: "Quotidian", pronunciation: "kwoh-TID-ee-un", pos: "adjective", definition: "Of or occurring every day; ordinary or commonplace.", example: "She found beauty in quotidian moments like morning coffee." },
    { word: "Raconteur", pronunciation: "rak-on-TUR", pos: "noun", definition: "A person who tells anecdotes in a skillful and amusing way.", example: "The dinner guest was a gifted raconteur who kept everyone entertained." },
    { word: "Sanguine", pronunciation: "SANG-gwin", pos: "adjective", definition: "Optimistic or positive, especially in an apparently bad situation.", example: "Despite the setbacks, she remained sanguine about their chances." },
    { word: "Temerity", pronunciation: "teh-MARE-ih-tee", pos: "noun", definition: "Excessive confidence or boldness; audacity.", example: "He had the temerity to criticize the chef's cooking in front of everyone." },

    // Week 32
    { word: "Urbane", pronunciation: "ur-BANE", pos: "adjective", definition: "Suave, courteous, and refined in manner.", example: "His urbane charm made him popular in diplomatic circles." },
    { word: "Vapid", pronunciation: "VAP-id", pos: "adjective", definition: "Offering nothing that is stimulating or challenging; bland.", example: "The vapid television show offered no intellectual content." },
    { word: "Wax", pronunciation: "WAKS", pos: "verb", definition: "To speak or write in a specified manner; to increase in vigor or intensity.", example: "He would often wax poetic about the beauty of nature." },
    { word: "Yaw", pronunciation: "YAW", pos: "verb/noun", definition: "To twist or oscillate about a vertical axis.", example: "The ship began to yaw in the heavy seas." },
    { word: "Zealot", pronunciation: "ZEL-ut", pos: "noun", definition: "A person who is fanatical and uncompromising in pursuit of religious, political, or other ideals.", example: "The zealot refused to consider any opposing viewpoints." },

    // Week 33
    { word: "Accrue", pronunciation: "uh-KROO", pos: "verb", definition: "To accumulate or receive over time.", example: "Interest accrues on the account monthly." },
    { word: "Bifurcate", pronunciation: "BYE-fur-kate", pos: "verb", definition: "To divide into two branches or forks.", example: "The path bifurcates at the old oak tree." },
    { word: "Confluence", pronunciation: "KON-floo-ens", pos: "noun", definition: "The junction of two rivers; an act or process of merging.", example: "The town was built at the confluence of two major waterways." },
    { word: "Dearth", pronunciation: "DERTH", pos: "noun", definition: "A scarcity or lack of something.", example: "There was a dearth of qualified candidates for the position." },
    { word: "Effete", pronunciation: "eh-FEET", pos: "adjective", definition: "No longer capable of effective action; weak or decadent.", example: "The effete aristocracy had lost touch with common people." },

    // Week 34
    { word: "Fecund", pronunciation: "FEK-und", pos: "adjective", definition: "Producing or capable of producing abundant vegetation or offspring; fruitful.", example: "The fecund soil produced bountiful harvests." },
    { word: "Gambit", pronunciation: "GAM-bit", pos: "noun", definition: "A device, action, or opening remark intended to gain an advantage.", example: "His offer to resign was just a gambit to gain more power." },
    { word: "Hermetic", pronunciation: "her-MET-ik", pos: "adjective", definition: "Completely airtight; protected from outside influences.", example: "The laboratory maintained hermetic conditions for the experiment." },
    { word: "Immutable", pronunciation: "ih-MYOO-tuh-bul", pos: "adjective", definition: "Unchanging over time or unable to be changed.", example: "The laws of physics are considered immutable." },
    { word: "Jettison", pronunciation: "JET-ih-sun", pos: "verb", definition: "To throw or drop something from an aircraft or ship; to abandon or discard.", example: "The company decided to jettison its unprofitable divisions." },

    // Week 35
    { word: "Kismet", pronunciation: "KIZ-met", pos: "noun", definition: "Destiny; fate.", example: "They believed it was kismet that brought them together." },
    { word: "Limpid", pronunciation: "LIM-pid", pos: "adjective", definition: "Completely clear and transparent; easy to understand.", example: "The limpid water of the mountain stream revealed every pebble." },
    { word: "Modicum", pronunciation: "MOD-ih-kum", pos: "noun", definition: "A small quantity of a particular thing.", example: "With a modicum of effort, you could pass the exam." },
    { word: "Nadir", pronunciation: "NAY-deer", pos: "noun", definition: "The lowest point in the fortunes of a person or organization.", example: "The company reached its nadir during the recession." },
    { word: "Ossify", pronunciation: "OSS-ih-fy", pos: "verb", definition: "To turn into bone or become rigid and inflexible.", example: "His opinions had ossified over the years." },

    // Week 36
    { word: "Penultimate", pronunciation: "peh-NUL-tih-mit", pos: "adjective", definition: "Last but one in a series of things; second to last.", example: "This is the penultimate episode of the season." },
    { word: "Quasi", pronunciation: "KWAY-zye", pos: "adjective/adverb", definition: "Seemingly; apparently but not really.", example: "The organization operates as a quasi-governmental agency." },
    { word: "Reticent", pronunciation: "RET-ih-sent", pos: "adjective", definition: "Not revealing one's thoughts or feelings readily.", example: "She was reticent about discussing her personal life." },
    { word: "Svelte", pronunciation: "SVELT", pos: "adjective", definition: "Slender and elegant.", example: "The dancer's svelte figure moved gracefully across the stage." },
    { word: "Toady", pronunciation: "TOE-dee", pos: "noun/verb", definition: "A person who behaves obsequiously to someone important.", example: "The office toady constantly flattered the boss." },

    // Week 37
    { word: "Usurp", pronunciation: "yoo-SURP", pos: "verb", definition: "To take a position of power or importance illegally or by force.", example: "The general attempted to usurp the throne." },
    { word: "Vex", pronunciation: "VEKS", pos: "verb", definition: "To make someone feel annoyed, frustrated, or worried.", example: "The problem continued to vex the researchers." },
    { word: "Wane", pronunciation: "WANE", pos: "verb", definition: "To decrease in vigor, power, or extent; to become weaker.", example: "Public interest in the scandal began to wane." },
    { word: "Yen", pronunciation: "YEN", pos: "noun", definition: "A longing or yearning.", example: "She had a yen for adventure and travel." },
    { word: "Zeitgeist", pronunciation: "ZYTE-gyst", pos: "noun", definition: "The defining spirit or mood of a particular period of history.", example: "The film captured the zeitgeist of the 1960s." },

    // Week 38
    { word: "Anodyne", pronunciation: "AN-uh-dyne", pos: "adjective/noun", definition: "Not likely to provoke dissent or offense; something that soothes.", example: "The politician's anodyne speech avoided all controversial topics." },
    { word: "Byzantine", pronunciation: "BIZ-un-teen", pos: "adjective", definition: "Excessively complicated and difficult to understand.", example: "The Byzantine regulations made compliance nearly impossible." },
    { word: "Catharsis", pronunciation: "kuh-THAR-sis", pos: "noun", definition: "The process of releasing strong or repressed emotions.", example: "Crying at the movie provided an emotional catharsis." },
    { word: "Denouement", pronunciation: "day-noo-MAHN", pos: "noun", definition: "The final part of a narrative in which matters are explained or resolved.", example: "The novel's denouement tied up all the loose ends." },
    { word: "Endemic", pronunciation: "en-DEM-ik", pos: "adjective", definition: "Regularly found among particular people or in a certain area.", example: "Malaria is endemic to tropical regions." },

    // Week 39
    { word: "Fortuitous", pronunciation: "for-TOO-ih-tus", pos: "adjective", definition: "Happening by accident or chance rather than design.", example: "Their fortuitous meeting at the airport led to a business partnership." },
    { word: "Gesticulate", pronunciation: "jess-TIK-yoo-late", pos: "verb", definition: "To use gestures, especially dramatic ones, instead of or in addition to speech.", example: "The excited speaker gesticulated wildly to emphasize her points." },
    { word: "Hapless", pronunciation: "HAP-less", pos: "adjective", definition: "Unfortunate; unlucky.", example: "The hapless victim was in the wrong place at the wrong time." },
    { word: "Inchoate", pronunciation: "in-KOH-it", pos: "adjective", definition: "Just begun and not fully formed or developed; rudimentary.", example: "His inchoate plans needed much more development." },
    { word: "Juxtaposition", pronunciation: "juck-stuh-puh-ZIH-shun", pos: "noun", definition: "The fact of two things being seen or placed close together with contrasting effect.", example: "The juxtaposition of wealth and poverty in the city was striking." },

    // Week 40
    { word: "Liminal", pronunciation: "LIM-in-ul", pos: "adjective", definition: "Relating to a transitional or initial stage; occupying a position at both sides of a boundary.", example: "The twilight hours occupy a liminal space between day and night." },
    { word: "Miscreant", pronunciation: "MISS-kree-ent", pos: "noun", definition: "A person who behaves badly or in a way that breaks the law.", example: "The miscreants vandalized several storefronts." },
    { word: "Noisome", pronunciation: "NOY-sum", pos: "adjective", definition: "Having an extremely offensive smell; disagreeable.", example: "The noisome odor from the factory upset residents." },
    { word: "Onerous", pronunciation: "OH-ner-us", pos: "adjective", definition: "Involving an amount of effort and difficulty that is oppressively burdensome.", example: "The onerous task of cataloging the entire library took months." },
    { word: "Penchant", pronunciation: "PEN-chent", pos: "noun", definition: "A strong or habitual liking for something.", example: "She had a penchant for collecting vintage jewelry." },

    // Week 41
    { word: "Quagmire", pronunciation: "KWAG-myer", pos: "noun", definition: "A soft boggy area of land; a complex or difficult situation.", example: "The project became a quagmire of bureaucratic delays." },
    { word: "Rancor", pronunciation: "RANG-ker", pos: "noun", definition: "Bitterness or resentfulness, especially when long-standing.", example: "Despite their divorce years ago, there was still rancor between them." },
    { word: "Soporific", pronunciation: "sop-or-IF-ik", pos: "adjective", definition: "Tending to induce drowsiness or sleep; tedious.", example: "The professor's soporific lecture put half the class to sleep." },
    { word: "Tendentious", pronunciation: "ten-DEN-shus", pos: "adjective", definition: "Expressing or intending to promote a particular cause or point of view.", example: "The documentary was criticized for its tendentious approach." },
    { word: "Untoward", pronunciation: "un-TORD", pos: "adjective", definition: "Unexpected and inappropriate or inconvenient.", example: "Nothing untoward happened during the ceremony." },

    // Week 42
    { word: "Variegated", pronunciation: "VAIR-ee-uh-gay-tid", pos: "adjective", definition: "Exhibiting different colors, especially as irregular patches or streaks.", example: "The variegated leaves of the plant were green and white." },
    { word: "Wan", pronunciation: "WON", pos: "adjective", definition: "Pale and appearing to be ill or exhausted.", example: "After her illness, she still looked wan and tired." },
    { word: "Xenophobia", pronunciation: "zen-uh-FOH-bee-uh", pos: "noun", definition: "Dislike of or prejudice against people from other countries.", example: "The rise in xenophobia worried human rights advocates." },
    { word: "Yeoman", pronunciation: "YOH-mun", pos: "noun", definition: "A man holding and cultivating a small landed estate; a diligent worker.", example: "He did yeoman's work organizing the event." },
    { word: "Zephyr", pronunciation: "ZEF-er", pos: "noun", definition: "A soft, gentle breeze.", example: "A pleasant zephyr cooled the warm afternoon." },

    // Week 43
    { word: "Acumen", pronunciation: "uh-KYOO-men", pos: "noun", definition: "The ability to make good judgments and quick decisions.", example: "Her business acumen helped the company thrive." },
    { word: "Bedlam", pronunciation: "BED-lum", pos: "noun", definition: "A scene of uproar and confusion.", example: "The office was bedlam on the day before the deadline." },
    { word: "Cogent", pronunciation: "KOH-jent", pos: "adjective", definition: "Clear, logical, and convincing.", example: "She made a cogent argument for changing the policy." },
    { word: "Desultory", pronunciation: "DESS-ul-tor-ee", pos: "adjective", definition: "Lacking a plan, purpose, or enthusiasm.", example: "Their desultory conversation wandered from topic to topic." },
    { word: "Erudite", pronunciation: "AIR-yoo-dyte", pos: "adjective", definition: "Having or showing great knowledge or learning.", example: "The erudite professor could discuss any subject in depth." },

    // Week 44
    { word: "Feckless", pronunciation: "FEK-less", pos: "adjective", definition: "Lacking initiative or strength of character; irresponsible.", example: "His feckless approach to work cost him the promotion." },
    { word: "Gauche", pronunciation: "GOHSH", pos: "adjective", definition: "Lacking ease or grace; unsophisticated and socially awkward.", example: "His gauche behavior at the formal dinner embarrassed his hosts." },
    { word: "Heuristic", pronunciation: "hyoo-RISS-tik", pos: "adjective", definition: "Enabling someone to discover or learn something for themselves.", example: "The teacher used a heuristic approach to help students solve problems." },
    { word: "Indefatigable", pronunciation: "in-deh-FAT-ih-guh-bul", pos: "adjective", definition: "Persisting tirelessly.", example: "Her indefatigable efforts finally paid off." },
    { word: "Jingoism", pronunciation: "JING-go-iz-um", pos: "noun", definition: "Extreme patriotism, especially in the form of aggressive foreign policy.", example: "The politician's jingoism appealed to nationalist sentiments." },

    // Week 45
    { word: "Knell", pronunciation: "NEL", pos: "noun", definition: "The sound of a bell, especially when rung solemnly for a death or funeral.", example: "The project's failure sounded the knell for similar initiatives." },
    { word: "Lissome", pronunciation: "LISS-um", pos: "adjective", definition: "Thin, supple, and graceful.", example: "The lissome gymnast performed with incredible flexibility." },
    { word: "Milieu", pronunciation: "mill-YOO", pos: "noun", definition: "A person's social environment.", example: "She felt out of place in the academic milieu." },
    { word: "Necromancy", pronunciation: "NEK-roh-man-see", pos: "noun", definition: "The supposed practice of communicating with the dead; sorcery.", example: "Medieval tales often featured wizards practicing necromancy." },
    { word: "Obeisance", pronunciation: "oh-BAY-sens", pos: "noun", definition: "Deferential respect; a gesture expressing this.", example: "The subjects made obeisance to the king." },

    // Week 46
    { word: "Pedestrian", pronunciation: "peh-DESS-tree-un", pos: "adjective", definition: "Lacking inspiration or excitement; dull.", example: "The pedestrian plot made the movie forgettable." },
    { word: "Quixotic", pronunciation: "kwik-SOT-ik", pos: "adjective", definition: "Exceedingly idealistic; unrealistic and impractical.", example: "His quixotic dream of world peace was admirable but naive." },
    { word: "Recumbent", pronunciation: "rih-KUM-bent", pos: "adjective", definition: "Lying down; reclining.", example: "The recumbent statue depicted the king at rest." },
    { word: "Supine", pronunciation: "soo-PINE", pos: "adjective", definition: "Lying face upward; failing to act as a result of moral weakness.", example: "He remained supine in the face of obvious injustice." },
    { word: "Trenchant", pronunciation: "TREN-chent", pos: "adjective", definition: "Vigorous or incisive in expression or style.", example: "Her trenchant criticism of the policy sparked debate." },

    // Week 47
    { word: "Uxorious", pronunciation: "uk-SOR-ee-us", pos: "adjective", definition: "Having or showing an excessive fondness for one's wife.", example: "His uxorious devotion was touching but sometimes excessive." },
    { word: "Vitiate", pronunciation: "VISH-ee-ate", pos: "verb", definition: "To spoil or impair the quality or efficiency of.", example: "The error vitiated the entire experiment." },
    { word: "Weltschmerz", pronunciation: "VELT-shmairts", pos: "noun", definition: "A feeling of melancholy and world-weariness.", example: "The poet's work was filled with weltschmerz about modern life." },
    { word: "Xyst", pronunciation: "ZIST", pos: "noun", definition: "A garden walk or terrace.", example: "They strolled along the xyst admiring the flowers." },
    { word: "Yokel", pronunciation: "YOH-kul", pos: "noun", definition: "An unsophisticated country person.", example: "The city dwellers unfairly dismissed him as a yokel." },

    // Week 48
    { word: "Abeyance", pronunciation: "uh-BAY-ens", pos: "noun", definition: "A state of temporary disuse or suspension.", example: "The project was held in abeyance pending further review." },
    { word: "Bonhomie", pronunciation: "bon-uh-MEE", pos: "noun", definition: "Cheerful friendliness; geniality.", example: "His bonhomie made him popular at social gatherings." },
    { word: "Comity", pronunciation: "KOM-ih-tee", pos: "noun", definition: "Courtesy and considerate behavior toward others.", example: "International comity required diplomatic respect between nations." },
    { word: "Dudgeon", pronunciation: "DUJ-un", pos: "noun", definition: "A feeling of offense or resentment.", example: "She left in high dudgeon after the insult." },
    { word: "Equivocate", pronunciation: "eh-KWIV-uh-kate", pos: "verb", definition: "To use ambiguous language to conceal the truth or avoid committing oneself.", example: "The witness equivocated when asked direct questions." },

    // Week 49
    { word: "Fatuous", pronunciation: "FATCH-oo-us", pos: "adjective", definition: "Silly and pointless.", example: "His fatuous comments added nothing to the discussion." },
    { word: "Gravitas", pronunciation: "GRAV-ih-tahs", pos: "noun", definition: "Dignity, seriousness, or solemnity of manner.", example: "The judge's gravitas commanded respect in the courtroom." },
    { word: "Halcyon", pronunciation: "HAL-see-on", pos: "adjective", definition: "Denoting a period of time that was idyllically happy and peaceful.", example: "She looked back on the halcyon days of her childhood." },
    { word: "Insouciant", pronunciation: "in-SOO-see-ent", pos: "adjective", definition: "Showing a casual lack of concern; indifferent.", example: "His insouciant attitude toward deadlines worried his manager." },
    { word: "Juggernaut", pronunciation: "JUG-er-nawt", pos: "noun", definition: "A huge, powerful, and overwhelming force or institution.", example: "The company became a juggernaut in the tech industry." },

    // Week 50
    { word: "Kenspeckle", pronunciation: "KEN-spek-ul", pos: "adjective", definition: "Conspicuous; easily recognizable.", example: "Her kenspeckle red coat made her easy to spot in a crowd." },
    { word: "Largesse", pronunciation: "lar-JESS", pos: "noun", definition: "Generosity in bestowing money or gifts upon others.", example: "The billionaire's largesse funded numerous charities." },
    { word: "Mordant", pronunciation: "MOR-dent", pos: "adjective", definition: "Having or showing a sharp or critical quality; biting.", example: "His mordant wit often offended sensitive people." },
    { word: "Neophyte", pronunciation: "NEE-oh-fyte", pos: "noun", definition: "A person who is new to a subject, skill, or belief.", example: "As a neophyte in chess, she had much to learn." },
    { word: "Opprobrious", pronunciation: "uh-PRO-bree-us", pos: "adjective", definition: "Expressing scorn or criticism; abusive.", example: "The opprobrious language in the letter shocked its recipient." },

    // Week 51
    { word: "Pedagogy", pronunciation: "PED-uh-go-jee", pos: "noun", definition: "The method and practice of teaching.", example: "Modern pedagogy emphasizes student-centered learning." },
    { word: "Quorum", pronunciation: "KWOR-um", pos: "noun", definition: "The minimum number of members required to be present for valid transaction of business.", example: "Without a quorum, the committee couldn't vote on the proposal." },
    { word: "Rescind", pronunciation: "rih-SIND", pos: "verb", definition: "To revoke, cancel, or repeal a law, order, or agreement.", example: "The board voted to rescind the controversial policy." },
    { word: "Sobriquet", pronunciation: "SO-brih-kay", pos: "noun", definition: "A person's nickname.", example: "The athlete earned the sobriquet 'The Flying Dutchman.'" },
    { word: "Transmogrify", pronunciation: "trans-MOG-rih-fy", pos: "verb", definition: "To transform in a surprising or magical manner.", example: "The fairy tale describes how the prince was transmogrified into a frog." },

    // Week 52
    { word: "Ukase", pronunciation: "yoo-KAZE", pos: "noun", definition: "An arbitrary or peremptory order or decree.", example: "The manager's ukase about working hours angered employees." },
    { word: "Verdure", pronunciation: "VUR-jer", pos: "noun", definition: "Lush green vegetation.", example: "The valley was covered in verdure after the spring rains." },
    { word: "Waggish", pronunciation: "WAG-ish", pos: "adjective", definition: "Playful in a humorous or mischievous manner.", example: "His waggish comments lightened the mood." },

    // Final days: Dec 29-31, 2026
    { word: "Syzygy", pronunciation: "SIZ-ih-jee", pos: "noun", definition: "A conjunction or opposition of celestial bodies; a pair of connected or corresponding things.", example: "The lunar eclipse occurred during a syzygy of the sun, earth, and moon." },
    { word: "Penumbra", pronunciation: "peh-NUM-bruh", pos: "noun", definition: "The partially shaded outer region of a shadow; a peripheral or indeterminate area.", example: "The theory operated in the penumbra between science and philosophy." }
];

// Service Worker Registration
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('service-worker.js')
            .then(registration => {
                console.log('ServiceWorker registered:', registration);
            })
            .catch(err => {
                console.log('ServiceWorker registration failed:', err);
            });
    });
}

// Get the word index based on the current date
function getWordIndex() {
    const startDate = new Date('2025-12-26'); // Friday, December 26, 2025
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const diffTime = today - startDate;
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    // Calculate weekday count (skip weekends)
    let weekdayCount = 0;
    for (let i = 0; i <= diffDays; i++) {
        const checkDate = new Date(startDate);
        checkDate.setDate(startDate.getDate() + i);
        const dayOfWeek = checkDate.getDay();
        if (dayOfWeek !== 0 && dayOfWeek !== 6) { // Not Sunday or Saturday
            weekdayCount++;
        }
    }

    return (weekdayCount - 1) % wordDatabase.length;
}

// Check if today is a weekend
function isWeekend() {
    const today = new Date();
    const dayOfWeek = today.getDay();
    return dayOfWeek === 0 || dayOfWeek === 6; // Sunday or Saturday
}

// Format date
function formatDate() {
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    return new Date().toLocaleDateString('en-US', options);
}

// Display word or weekend message
function displayWord() {
    const dateElement = document.getElementById('currentDate');
    dateElement.textContent = formatDate();

    if (isWeekend()) {
        document.getElementById('wordContent').style.display = 'none';
        document.getElementById('weekendMessage').style.display = 'block';
    } else {
        const wordIndex = getWordIndex();
        const wordData = wordDatabase[wordIndex];

        document.getElementById('word').textContent = wordData.word;
        document.getElementById('pronunciation').textContent = wordData.pronunciation;
        document.getElementById('partOfSpeech').textContent = wordData.pos;
        document.getElementById('definition').textContent = wordData.definition;
        document.getElementById('example').textContent = `"${wordData.example}"`;

        document.getElementById('wordContent').style.display = 'block';
        document.getElementById('weekendMessage').style.display = 'none';
    }
}

// ============================================================================
// NOTIFICATION CODE - COMMENTED OUT UNTIL BACKEND SERVER IS SET UP
// ============================================================================
// Note: Push notifications require a backend server to send notifications
// when the app is closed. Uncomment this code once the server is ready.
// ============================================================================

/*
async function requestNotificationPermission() {
    if (!('Notification' in window)) {
        alert('This browser does not support notifications');
        return;
    }

    if (!('serviceWorker' in navigator)) {
        alert('This browser does not support service workers');
        return;
    }

    const permission = await Notification.requestPermission();

    if (permission === 'granted') {
        await subscribeUserToPush();
        document.getElementById('notificationPrompt').style.display = 'none';
        localStorage.setItem('notificationsEnabled', 'true');

        // Schedule daily notifications
        scheduleDailyNotification();
    }
}

async function subscribeUserToPush() {
    const registration = await navigator.serviceWorker.ready;

    try {
        const subscription = await registration.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: urlBase64ToUint8Array('BEl62iUYgUivxIkv69yViEuiBIa-Ib9-SkvMeAtA3LFgDSKidOrChZlNfmYzxaPAFy-KVDPqcDh0Y5GlPmCmLNE')
        });

        console.log('User is subscribed:', subscription);
    } catch (err) {
        console.log('Failed to subscribe the user:', err);
    }
}

function urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
        .replace(/\-/g, '+')
        .replace(/_/g, '/');

    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
        outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
}

function scheduleDailyNotification() {
    // Check every hour if it's 9 AM on a weekday
    setInterval(() => {
        const now = new Date();
        const hour = now.getHours();
        const day = now.getDay();

        // If it's 9 AM on a weekday (1-5)
        if (hour === 9 && day >= 1 && day <= 5) {
            showNotification();
        }
    }, 60 * 60 * 1000); // Check every hour
}

async function showNotification() {
    const permission = Notification.permission;

    if (permission === 'granted') {
        const wordIndex = getWordIndex();
        const wordData = wordDatabase[wordIndex];

        const registration = await navigator.serviceWorker.ready;
        registration.showNotification('Word of the Day', {
            body: `Today's word: ${wordData.word} - ${wordData.definition}`,
            icon: 'icons/icon-192.png',
            badge: 'icons/icon-192.png',
            vibrate: [200, 100, 200],
            tag: 'word-of-the-day',
            requireInteraction: false
        });
    }
}
*/

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    displayWord();

    // ============================================================================
    // NOTIFICATION INITIALIZATION - COMMENTED OUT UNTIL BACKEND SERVER IS SET UP
    // ============================================================================
    /*
    // Check notification permission status
    if ('Notification' in window && Notification.permission === 'default') {
        document.getElementById('notificationPrompt').style.display = 'block';
    } else if (Notification.permission === 'granted') {
        document.getElementById('notificationPrompt').style.display = 'none';
        scheduleDailyNotification();
    } else {
        document.getElementById('notificationPrompt').style.display = 'none';
    }

    // Add event listener for notification button
    const notifButton = document.getElementById('enableNotifications');
    if (notifButton) {
        notifButton.addEventListener('click', requestNotificationPermission);
    }
    */
});
