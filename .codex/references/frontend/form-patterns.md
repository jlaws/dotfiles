
# Form Patterns

## Stack Decision

| Need | Use |
|------|-----|
| Client-side React forms | React Hook Form + Zod |
| Next.js server mutations | Server Actions + useActionState |
| Simple 1-2 field forms | Uncontrolled + native validation |
| Complex multi-step | React Hook Form + state machine |

## React Hook Form + Zod

Use `zodResolver(schema)` with `useForm`. Register inputs with `{...register('field')}`. Display `errors.field.message` with `role='alert'` and `aria-describedby`.

### Code Example: RHF + Zod Login Form

```typescript
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(8, 'Password must be at least 8 chars'),
});

type LoginForm = z.infer<typeof loginSchema>;

export function LoginForm() {
  const { register, handleSubmit, formState: { errors } } = useForm<LoginForm>({
    resolver: zodResolver(loginSchema),
    mode: 'onBlur',
  });

  const onSubmit = (data: LoginForm) => console.log(data);

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div>
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          {...register('email')}
          aria-describedby={errors.email ? 'email-error' : undefined}
        />
        {errors.email && (
          <p id="email-error" role="alert" className="text-red-600">
            {errors.email.message}
          </p>
        )}
      </div>

      <div>
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          {...register('password')}
          aria-describedby={errors.password ? 'password-error' : undefined}
        />
        {errors.password && (
          <p id="password-error" role="alert" className="text-red-600">
            {errors.password.message}
          </p>
        )}
      </div>

      <button type="submit">Sign In</button>
    </form>
  );
}
```

## Multi-Step Wizard

Track current step in state. Merge step data on each submit. Each step is its own RHF instance with `defaultValues` from accumulated data.

### Code Example: Multi-Step Wizard with Validation

```typescript
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

// Step 1: Personal info
const step1Schema = z.object({
  firstName: z.string().min(1, 'Required'),
  lastName: z.string().min(1, 'Required'),
});

// Step 2: Address
const step2Schema = z.object({
  street: z.string().min(1, 'Required'),
  city: z.string().min(1, 'Required'),
  zipCode: z.string().regex(/^\d{5}$/, 'Invalid ZIP'),
});

type Step1Data = z.infer<typeof step1Schema>;
type Step2Data = z.infer<typeof step2Schema>;

export function RegistrationWizard() {
  const [step, setStep] = useState(1);
  const [accumulatedData, setAccumulatedData] = useState<Partial<Step1Data & Step2Data>>({});

  const form1 = useForm<Step1Data>({
    resolver: zodResolver(step1Schema),
    defaultValues: accumulatedData,
  });

  const handleStep1Submit = (data: Step1Data) => {
    setAccumulatedData(prev => ({ ...prev, ...data }));
    setStep(2);
  };

  const handleStep2Submit = (data: Step2Data) => {
    const finalData = { ...accumulatedData, ...data };
    console.log('Submit:', finalData);
  };

  if (step === 1) {
    return (
      <form onSubmit={form1.handleSubmit(handleStep1Submit)}>
        <h2>Step 1: Personal Info</h2>
        <input {...form1.register('firstName')} placeholder="First name" />
        <input {...form1.register('lastName')} placeholder="Last name" />
        {form1.formState.errors.firstName && (
          <p role="alert">{form1.formState.errors.firstName.message}</p>
        )}
        <button type="submit">Next</button>
      </form>
    );
  }

  const form2 = useForm<Step2Data>({
    resolver: zodResolver(step2Schema),
    defaultValues: accumulatedData,
  });

  return (
    <form onSubmit={form2.handleSubmit(handleStep2Submit)}>
      <h2>Step 2: Address</h2>
      <input {...form2.register('street')} placeholder="Street" />
      <input {...form2.register('city')} placeholder="City" />
      <input {...form2.register('zipCode')} placeholder="ZIP Code" />
      {form2.formState.errors.zipCode && (
        <p role="alert">{form2.formState.errors.zipCode.message}</p>
      )}
      <button type="button" onClick={() => setStep(1)}>Back</button>
      <button type="submit">Submit</button>
    </form>
  );
}
```

## Server Actions (Next.js)

Mark action files `'use server'`. Parse FormData with Zod's `safeParse`. Return `{ errors }` on failure. Client uses `useActionState(action, initialState)` returning `[state, formAction, isPending]`.

### Code Example: Server Action with useActionState

**Action (app/actions.ts):**
```typescript
'use server';

import { z } from 'zod';

const createPostSchema = z.object({
  title: z.string().min(1, 'Title required').max(100),
  content: z.string().min(10, 'Content must be at least 10 chars'),
});

type ActionState = {
  errors?: Record<string, string>;
  success?: boolean;
};

export async function createPost(prevState: ActionState, formData: FormData): Promise<ActionState> {
  const parsed = createPostSchema.safeParse({
    title: formData.get('title'),
    content: formData.get('content'),
  });

  if (!parsed.success) {
    return {
      errors: parsed.error.flatten().fieldErrors as Record<string, string>,
    };
  }

  // Save to database
  console.log('Saving post:', parsed.data);

  return { success: true };
}
```

**Client Component (app/components/CreatePostForm.tsx):**
```typescript
'use client';

import { useActionState } from 'react';
import { createPost } from '@/app/actions';

export function CreatePostForm() {
  const [state, formAction, isPending] = useActionState(createPost, { errors: {} });

  return (
    <form action={formAction}>
      <div>
        <label htmlFor="title">Title</label>
        <input id="title" name="title" required />
        {state.errors?.title && (
          <p role="alert" className="text-red-600">{state.errors.title}</p>
        )}
      </div>

      <div>
        <label htmlFor="content">Content</label>
        <textarea id="content" name="content" required />
        {state.errors?.content && (
          <p role="alert" className="text-red-600">{state.errors.content}</p>
        )}
      </div>

      <button type="submit" disabled={isPending}>
        {isPending ? 'Posting...' : 'Create Post'}
      </button>

      {state.success && <p className="text-green-600">Post created!</p>}
    </form>
  );
}
```

## File Upload

Use `useRef` for hidden file input. Handle drag-and-drop via `onDrop`/`onDragOver`. Transfer dropped files to input via `new DataTransfer()` + `dt.items.add(file)` for form submission.

### Code Example: Drag-and-Drop File Upload

```typescript
import { useRef, useState } from 'react';

export function FileUploadDropZone() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [dragActive, setDragActive] = useState(false);

  const handleFiles = (fileList: FileList) => {
    const newFiles = Array.from(fileList);
    setFiles(prev => [...prev, ...newFiles]);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = () => {
    setDragActive(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  };

  const handleRemoveFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div>
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`border-2 border-dashed rounded p-8 text-center cursor-pointer ${
          dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
        }`}
        onClick={() => inputRef.current?.click()}
      >
        <p className="text-gray-600">Drag files here or click to select</p>
      </div>

      <input
        ref={inputRef}
        type="file"
        multiple
        onChange={handleInputChange}
        className="hidden"
        aria-label="File upload"
      />

      {files.length > 0 && (
        <div className="mt-4">
          <h3 className="font-semibold">Selected Files ({files.length}):</h3>
          <ul>
            {files.map((file, idx) => (
              <li key={idx} className="flex justify-between items-center p-2 bg-gray-100 rounded mt-1">
                <span>{file.name}</span>
                <button
                  type="button"
                  onClick={() => handleRemoveFile(idx)}
                  className="text-red-600"
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

## Validation UX

Use `mode: 'onBlur'` for inline validation. Debounce async validators (username availability). Prefer inline-on-blur for most fields, on-submit for short forms.

## Accessibility Checklist

- Every `<input>` has a `<label>` with matching `htmlFor`/`id`
- Error messages use `role="alert"` and link via `aria-describedby`
- Required fields use `aria-required="true"` (not just `required` attribute)
- Focus the first error field on submit failure
- Disabled submit buttons still explain why (tooltip or adjacent text)

## Anti-Patterns

- **Controlled everything**: RHF is uncontrolled by default for perf; only wrap Select/DatePicker in `Controller` if they don't support `ref`
- **Async validation without debounce**: Always debounce username/email checks; server will rate-limit
- **Storing server errors in client state**: Use `useActionState` pattern; don't manually manage Redux for form errors
- **FormData with nested objects**: FormData is flat; encode nested data as JSON strings or use URLSearchParams with arrays

## Gotchas

- **Controlled vs uncontrolled**: React Hook Form is uncontrolled by default; `Controller` wraps controlled components (Select, DatePicker). Don't mix `register` with `value` prop
- **FormData with checkboxes**: unchecked checkboxes are absent from FormData, not `false` -- parse with `formData.get('field') === 'on'`
- **File input reset**: setting `input.value = ''` is the only way to clear; React state won't do it
- **Server Action errors**: `useActionState` replaces `useFormState` in React 19; always return structured error objects, not thrown errors
- **Zod `.transform()`**: transforms run after validation; `z.coerce.number()` for FormData string-to-number

## Cross-References

- **frontend:react-state-management** -- managing form state alongside global state
- **frontend:nextjs-app-router-patterns** -- server actions, revalidation, and progressive enhancement
- **languages:pydantic-and-data-validation** -- server-side schema validation mirroring Zod patterns
