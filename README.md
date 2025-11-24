# Connect4 Tournament — README

Este repositorio contiene el entorno completo para ejecutar el **torneo de agentes de Connect4**.  
Incluye grupos de políticas (agents), el main ejecutable del torneo y las herramientas internas necesarias.

---

## Estructura del Proyecto

tournament/
│
├── groups/
│   ├── group_A/ El agente principal
│   ├── policy.py 
│   └── brain.pkl.gz # Archivo comprimido donde tu agente guarda el conocimiento
│   │
│   ├── group_B/  Primera versión del agente
│   │ ├── init.py
│   │ └── policy.py 
│   │
│   ├── group_C/  Agente aleatorio
│   │ ├── init.py
│   │ └── policy.py 
│
├── connect4/ # Motor del juego
│ ├── init.py
│ ├── dtos.py
│ ├── policy.py
│ ├── utils.py
│ └──  enviroment_state.py
│
├── main.py 
├── tournament.py 
└── README.md (este documento)



##  ¿Cómo ejecutar este torneo?

Para ejecutar este tonero, se debe dirigir al archivo Main.py y ejecutarlo, allí el torneo iniciará con los agentes en las carpetas.
El agente principal que está en GroupA va a generar su archivo brain_optimized.pkl.gz en su misma carpeta y guardará su conocimiento.
No hay que hacer configuracione adicionales, esto ya permite que se pueda jugar e iterar cuantas veces se desee.

