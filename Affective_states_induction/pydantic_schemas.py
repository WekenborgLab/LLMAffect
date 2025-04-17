from typing import Literal
from pydantic import BaseModel, Field


class PANAS(BaseModel):
    """Positive and Negative Affect Schedule (PANAS) measuring emotions at the present moment"""
    interested: Literal[0, 1, 2, 3, 4]
    distressed: Literal[0, 1, 2, 3, 4]
    excited: Literal[0, 1, 2, 3, 4]
    upset: Literal[0, 1, 2, 3, 4]
    strong: Literal[0, 1, 2, 3, 4]
    guilty: Literal[0, 1, 2, 3, 4]
    scared: Literal[0, 1, 2, 3, 4]
    hostile: Literal[0, 1, 2, 3, 4]
    enthusiastic: Literal[0, 1, 2, 3, 4]
    proud: Literal[0, 1, 2, 3, 4]
    irritable: Literal[0, 1, 2, 3, 4]
    alert: Literal[0, 1, 2, 3, 4]
    ashamed: Literal[0, 1, 2, 3, 4]
    inspired: Literal[0, 1, 2, 3, 4]
    nervous: Literal[0, 1, 2, 3, 4]
    determined: Literal[0, 1, 2, 3, 4]
    attentive: Literal[0, 1, 2, 3, 4]
    confused: Literal[0, 1, 2, 3, 4]
    active: Literal[0, 1, 2, 3, 4]
    afraid: Literal[0, 1, 2, 3, 4]


class StateAnxiety(BaseModel):
    """State Anxiety Inventory measuring current levels of anxiety"""
    calm: Literal[1, 2, 3, 4]
    secure: Literal[1, 2, 3, 4]
    tense: Literal[1, 2, 3, 4]
    troubled: Literal[1, 2, 3, 4]
    at_ease: Literal[1, 2, 3, 4]
    excited: Literal[1, 2, 3, 4]
    worried_wrong: Literal[1, 2, 3, 4]
    rested: Literal[1, 2, 3, 4]
    anxious: Literal[1, 2, 3, 4]
    comfortable: Literal[1, 2, 3, 4]
    confident: Literal[1, 2, 3, 4]
    nervous: Literal[1, 2, 3, 4]
    jittery: Literal[1, 2, 3, 4]
    strained: Literal[1, 2, 3, 4]
    relaxed: Literal[1, 2, 3, 4]
    content: Literal[1, 2, 3, 4]
    worried: Literal[1, 2, 3, 4]
    overstimulated: Literal[1, 2, 3, 4]
    happy: Literal[1, 2, 3, 4]
    cheerful: Literal[1, 2, 3, 4]


class VASScales(BaseModel):
    """Visual Analogue Scales measuring emotional intensity"""
    stress: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                   41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
                   61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 
                   81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    fear: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
                 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 
                 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    sadness: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                      41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
                      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 
                      81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    disgust: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
                    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 
                    81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anger: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                  41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
                  61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 
                  81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    worry: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                  41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
                  61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 
                  81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]


class EmotionalStateAssessment(BaseModel):
    """A comprehensive assessment of current emotional state using validated psychological measures"""
    panas: PANAS = Field(..., description="Positive and Negative Affect Schedule (PANAS) measuring emotions at the present moment")
    state_anxiety: StateAnxiety = Field(..., description="State Anxiety Inventory measuring current levels of anxiety")
    vas_scales: VASScales = Field(..., description="Visual Analogue Scales measuring emotional intensity")
