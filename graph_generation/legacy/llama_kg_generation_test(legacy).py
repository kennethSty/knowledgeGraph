from utils import kg_utils
from utils.kg_utils import  med_llama_prompt



text = """Erste, in der Steinzeit entstandene Krankheitstheorien waren in Bezug auf in den Körper eingedrungene Fremdkörper oder eine von außen entstandene Einwirkung rein empirisch, woraus sich dann eine Fremdkörper- und Emanationspathologie sowie eine „präanimistische“ Medizin entwickelten, deren Ziel es war, die Fremdkörper auszutreiben. Im Rahmen einer animistischen Weltanschauung schloss sich dann die Personifikation des Fremdkörpers (als Krankheitsdämon bzw. Besessenheit) an und eine übernatürliche Emanation als Zauber (als Dämonenvertreibung bzw. Gegenzauber). Mit Erreichen einer höheren Kulturstufe erschien die Krankheit als Strafe Gottes (Theurgische Pathologie) und die Heilhandlung erfolgte als Kulthandlung durch den Priesterarzt. Im alten Judentum wurde Krankheit als seelische und körperliche Läuterung des Menschen angesehen. Die im antiken Griechenland entstandene hippokratische Medizin definierte Krankheit als Störung im Säftehaushalt des Körpers und schuf damit die Krankheitstheorie der vor allem in der mittelalterlichen Schulmedizin grundlegenden Humoralpathologie.
Mit Beginn der Neuzeit wurde Krankheit zunehmend als Störung des Organismus begriffen.
Die Einordnung, das Maß der „Normalität“ überschreitender Veränderungen eines Menschen, hängt stark von der Kultur und der Epoche ab. So war Adipositas (Fettleibigkeit) in der Renaissance ein Status-Symbol, heutzutage wird sie allgemein als krankhaft betrachtet.
Dass bestimmte chemische Elemente die Grundbestandteile von lebenden Organismen sind, war zu Beginn des 19. Jahrhunderts dann auch Teil eines medizinischen Konzeptes des französischen, an die Entwicklungen in der Chemie seiner Zeit anknüpfenden Arztes Jean Baptiste Thimotée Baumes (1756–1828). Nach dessen 1806 publizierten Ansichten reagieren diese Elemente entsprechend ihrer chemischen Affinität im Körper. Krankheiten seien demnach neben Störungen im Wärmehaushalt und Wasserhaushalt auch solche des Stickstoffhaushalts oder Phosphorhaushalts.
Erste umfangreiche Untersuchungen zur Vererbbarkeit von Krankheit und von Krankheitsdisposition erfolgten ab dem Ende des 19. Jahrhunderts, etwa durch Rudolf Virchow, Johannes
Orth, Ernst Ziegler, Paul Baumgarten, Felix Victor Birch-Hirschfeld und Otto Lubarsch. Zur Übertragbarkeit von Krankheitserregern, Giften (und Immunstoffen) forschten um diese Zeit Erich Werner, Hermann Merkel, Adolf Gottstein und andere."""

generator_chain = kg_utils.init_llama_chain(model="llama3", prompt=med_llama_prompt)
result = generator_chain.invoke({"input": text})

print(result)
