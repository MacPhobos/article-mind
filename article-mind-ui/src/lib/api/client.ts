/**
 * Base API client for article-mind-service
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface ApiError {
	error: {
		code: string;
		message: string;
		details?: unknown;
	};
}

export class ApiClient {
	private baseUrl: string;

	constructor(baseUrl: string = API_BASE_URL) {
		this.baseUrl = baseUrl;
	}

	async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
		const url = `${this.baseUrl}${endpoint}`;

		try {
			const response = await fetch(url, {
				...options,
				headers: {
					'Content-Type': 'application/json',
					...options?.headers
				}
			});

			if (!response.ok) {
				const error: ApiError = await response.json();
				throw new Error(error.error.message || 'API request failed');
			}

			return response.json();
		} catch (error) {
			console.error('API request failed:', error);
			throw error;
		}
	}

	async get<T>(endpoint: string): Promise<T> {
		return this.fetch<T>(endpoint, { method: 'GET' });
	}

	async post<T>(endpoint: string, data: unknown): Promise<T> {
		return this.fetch<T>(endpoint, {
			method: 'POST',
			body: JSON.stringify(data)
		});
	}

	async patch<T>(endpoint: string, data: unknown): Promise<T> {
		return this.fetch<T>(endpoint, {
			method: 'PATCH',
			body: JSON.stringify(data)
		});
	}

	async delete<T>(endpoint: string): Promise<T> {
		return this.fetch<T>(endpoint, { method: 'DELETE' });
	}
}

export const apiClient = new ApiClient();
