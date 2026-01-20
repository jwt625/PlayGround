// Type definitions for SLM-Guessr

export interface Sample {
	id: string;
	level: number;
	category: string;
	name: string;
	description: string;
	phase_gif: string;
	intensity_gif: string;
	parameters?: Record<string, unknown>;
}

export interface SampleManifest {
	samples: Sample[];
	generated_at: string;
	version: string;
}

export type QuizMode = 'phase-to-intensity' | 'intensity-to-phase';

export type Difficulty = 'easy' | 'medium' | 'hard';

export interface QuizQuestion {
	sample: Sample;
	mode: QuizMode;
	options: Sample[];
	correctIndex: number;
}

export interface QuizState {
	mode: QuizMode;
	difficulty: Difficulty;
	currentQuestion: number;
	totalQuestions: number;
	score: number;
	streak: number;
	answers: boolean[];
}

export interface HighScore {
	score: number;
	mode: QuizMode;
	difficulty: Difficulty;
	date: string;
}

